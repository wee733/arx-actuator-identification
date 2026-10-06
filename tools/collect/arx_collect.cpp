// ARX R5 actuator-identification collector.
//
// Reproduces the deployed command chain so captures identify exactly what a
// policy drives on the robot:
//
//   policy-rate thread (default 50 Hz)  set_target(q)          ─┐
//   1 kHz thread                         smooth += a(target-smooth) -> setJointPositions
//   sampling thread (default 100 Hz)     log raw/smooth command, position, velocity, current
//
// Profiles:
//   chirp      log-frequency sweep up and down on one joint, then a high-frequency hold
//   multisine  five incommensurate sines on one joint
//   step       random zero-order-hold steps on one joint (held-out test)
//   static     visit a list of poses; settle, then record (current-to-torque calibration)
//
// Only the ARX SDK (arx::r5::InterfacesThread) is required. On exit or abort the
// arm returns to the start pose and the SDK is switched to gravity compensation.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <boost/program_options.hpp>

#include "arx_r5_src/interfaces/InterfacesThread.hpp"

namespace po = boost::program_options;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kDof = 6;
using Joints = std::array<double, kDof>;

// Conservative joint limits (rad) for R5a joints 1-6.
constexpr Joints kJointLower{-3.14, 0.0, 0.0, -1.3, -1.57, -2.1};
constexpr Joints kJointUpper{2.6, 3.0, 3.14, 1.3, 1.57, 2.1};
// Average joint speed allowed when moving between static poses.
constexpr double kMaxStaticMoveSpeed = 0.3;

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

Joints to_joints(const std::vector<double>& values) {
    Joints out{};
    for (int j = 0; j < kDof && j < static_cast<int>(values.size()); ++j) out[j] = values[j];
    return out;
}

Joints parse_joints(const std::string& text) {
    std::vector<double> values;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) values.push_back(std::stod(token));
    if (values.size() != kDof) throw std::invalid_argument("expected 6 comma-separated values: " + text);
    return to_joints(values);
}

std::string format_joints(const Joints& q) {
    std::ostringstream out;
    out << "[";
    for (int j = 0; j < kDof; ++j) out << q[j] << (j + 1 < kDof ? ", " : "]");
    return out.str();
}

// The deployed actuator command path: a target written at policy rate and an
// exponential moving average stepped at 1 kHz before every CAN position command.
class SmoothedArm {
public:
    SmoothedArm(const std::string& urdf, const std::string& can, double alpha)
        : arm_(urdf, can, 0), alpha_(alpha) {
        arm_.setArmStatus(arx::r5::InterfacesThread::POSITION_CONTROL);
        target_ = smooth_ = last_command_ = position();
    }

    ~SmoothedArm() { arm_.setArmStatus(arx::r5::InterfacesThread::G_COMPENSATION); }

    void set_target(const Joints& q) {
        std::lock_guard<std::mutex> lock(mutex_);
        target_ = q;
    }

    // Must be called at exactly 1 kHz by a single thread.
    void step_smooth() {
        Joints target;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            target = target_;
        }
        std::vector<double> command(kDof);
        for (int j = 0; j < kDof; ++j) {
            smooth_[j] += alpha_ * (target[j] - smooth_[j]);
            command[j] = smooth_[j];
        }
        arm_.setJointPositions(command);
        std::lock_guard<std::mutex> lock(mutex_);
        last_command_ = smooth_;
    }

    Joints last_command() {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_command_;
    }

    Joints position() { return to_joints(arm_.getJointPositons()); }
    Joints velocity() { return to_joints(arm_.getJointVelocities()); }

    Joints current() {
        auto values = arm_.getJointCurrent();
        if (static_cast<int>(values.size()) < kDof) {
            Joints missing;
            missing.fill(std::numeric_limits<double>::quiet_NaN());
            return missing;
        }
        return to_joints(values);
    }

private:
    arx::r5::InterfacesThread arm_;
    double alpha_;
    std::mutex mutex_;
    Joints target_{};
    Joints smooth_{};
    Joints last_command_{};
};

struct Excitation {
    std::string profile = "chirp";
    int joint = 1;  // zero-based
    double amplitude = 0.03;
    double f0 = 0.1;
    double f1 = 3.0;
    double duration = 30.0;
    double high_freq_hold = 2.0;
    double step_period = 1.5;
    int step_count = 20;
    unsigned step_seed = 0;
    std::vector<Joints> static_poses;
    double static_move = 3.0;
    double static_settle = 3.0;
    double static_record = 3.0;
};

// Offset of the excited joint for chirp/multisine/step; done marks the end.
double excitation_offset(const Excitation& c, double t, bool& done) {
    done = false;
    if (c.profile == "chirp") {
        const double total = 2.0 * c.duration + c.high_freq_hold;
        if (t >= total) { done = true; return 0.0; }
        const double k = std::log(c.f1 / c.f0);
        auto phase = [&](double tt) {
            if (std::abs(k) < 1e-9) return 2.0 * M_PI * c.f0 * tt;
            return 2.0 * M_PI * c.f0 * c.duration / k * (std::exp(k * tt / c.duration) - 1.0);
        };
        double p;
        if (t < c.duration) {
            p = phase(t);
        } else if (t < 2.0 * c.duration) {
            // Time-mirrored down sweep keeps the phase continuous.
            p = 2.0 * phase(c.duration) - phase(2.0 * c.duration - t);
        } else {
            p = 2.0 * phase(c.duration) + 2.0 * M_PI * c.f1 * (t - 2.0 * c.duration);
        }
        return c.amplitude * std::sin(p);
    }
    if (c.profile == "multisine") {
        if (t >= c.duration) { done = true; return 0.0; }
        const double freqs[5] = {0.2, 0.53, 0.91, 1.7, 2.6};
        double sum = 0.0;
        for (double f : freqs) sum += std::sin(2.0 * M_PI * f * t);
        return c.amplitude / 5.0 * sum;
    }
    if (c.profile == "step") {
        if (t >= c.step_period * c.step_count) { done = true; return 0.0; }
        const int index = static_cast<int>(t / c.step_period);
        const unsigned h = static_cast<unsigned>(index) * 2654435761u + c.step_seed * 40503u + 12345u;
        const double u = static_cast<double>(h % 10000u) / 10000.0;
        return c.amplitude * (2.0 * u - 1.0);
    }
    done = true;
    return 0.0;
}

// Static profile: linear move from the previous pose, settle, then record.
Joints static_target(const Excitation& c, const Joints& start, double t, bool& recording,
                     bool& done, int& pose_index) {
    const double segment = c.static_move + c.static_settle + c.static_record;
    pose_index = static_cast<int>(t / segment);
    recording = false;
    done = pose_index >= static_cast<int>(c.static_poses.size());
    if (done) return start;
    const double local = t - pose_index * segment;
    const Joints& from = pose_index == 0 ? start : c.static_poses[pose_index - 1];
    const Joints& to = c.static_poses[pose_index];
    const double s = std::min(1.0, local / c.static_move);
    Joints q{};
    for (int j = 0; j < kDof; ++j) q[j] = (1.0 - s) * from[j] + s * to[j];
    recording = local >= c.static_move + c.static_settle;
    return q;
}

struct Options {
    std::string urdf, can, out;
    double ema_alpha = 0.05;
    double sample_hz = 100.0;
    double set_target_hz = 50.0;
    double ramp_s = 2.0;
    double return_s = 1.0;
    double max_track_error = 0.35;
    Joints start_pose{0.0, 1.0, 0.8, 0.0, 0.0, 0.0};
    Excitation excitation;
};

bool within_limits(const Joints& q) {
    for (int j = 0; j < kDof; ++j)
        if (q[j] < kJointLower[j] || q[j] > kJointUpper[j]) return false;
    return true;
}

// All checks run before the hardware is opened.
std::string validate(const Options& o) {
    const Excitation& c = o.excitation;
    if (!(o.ema_alpha > 0.0 && o.ema_alpha < 1.0)) return "ema-alpha must lie in (0, 1)";
    if (!(o.sample_hz > 0.0 && o.set_target_hz > 0.0 && o.set_target_hz <= o.sample_hz))
        return "need 0 < set-target-hz <= sample-hz";
    if (o.ramp_s < 0.0 || o.return_s < 0.0) return "ramp-s and return-s must be >= 0";
    if (!within_limits(o.start_pose)) return "start pose outside joint limits";
    if (c.profile == "static") {
        if (c.static_poses.empty()) return "profile static needs --static-poses";
        if (!(c.static_move > 0.0 && c.static_settle >= 0.0 && c.static_record > 0.0))
            return "static-move and static-record must be > 0, static-settle >= 0";
        const Joints* from = &o.start_pose;
        for (const Joints& pose : c.static_poses) {
            if (!within_limits(pose)) return "static pose outside joint limits: " + format_joints(pose);
            for (int j = 0; j < kDof; ++j)
                if (std::abs(pose[j] - (*from)[j]) / c.static_move > kMaxStaticMoveSpeed)
                    return "static move too fast for joint " + std::to_string(j) + "; increase --static-move";
            from = &pose;
        }
        return "";
    }
    if (c.profile != "chirp" && c.profile != "multisine" && c.profile != "step")
        return "profile must be chirp, multisine, step or static";
    if (c.joint < 0 || c.joint >= kDof) return "joint must be 0..5";
    if (!(c.amplitude > 0.0 && c.duration > 0.0 && c.f0 > 0.0 && c.f1 > 0.0))
        return "amp, dur, f0 and f1 must be > 0";
    if (o.start_pose[c.joint] - c.amplitude < kJointLower[c.joint] ||
        o.start_pose[c.joint] + c.amplitude > kJointUpper[c.joint])
        return "excitation range exceeds the joint limit";
    std::ofstream probe(o.out, std::ios::app);
    if (!probe) return "cannot write " + o.out;
    return "";
}

void write_meta(const Options& o, long rows) {
    const Excitation& c = o.excitation;
    std::ofstream m(o.out + ".meta.yaml");
    m << "schema: arx_sysid_capture_v2\n";
    m << "control_rate_hz: " << o.sample_hz << "\n";
    m << "set_target_hz: " << o.set_target_hz << "\n";
    m << "ema_alpha: " << o.ema_alpha << "\n";
    m << "ema_rate_hz: 1000\n";
    m << "profile: " << c.profile << "\n";
    m << "active_joint: " << (c.profile == "static" ? -1 : c.joint) << "\n";
    m << "amplitude_rad: " << c.amplitude << "\n";
    m << "freq_start_hz: " << c.f0 << "\n";
    m << "freq_end_hz: " << c.f1 << "\n";
    m << "duration_s: " << c.duration << "\n";
    m << "step_period_s: " << c.step_period << "\n";
    m << "step_count: " << c.step_count << "\n";
    m << "default_joint_pos: " << format_joints(o.start_pose) << "\n";
    if (c.profile == "static") {
        m << "static_move_s: " << c.static_move << "\n";
        m << "static_settle_s: " << c.static_settle << "\n";
        m << "static_record_s: " << c.static_record << "\n";
        m << "static_poses:\n";
        for (const Joints& pose : c.static_poses) m << "  - " << format_joints(pose) << "\n";
    }
    m << "n_rows: " << rows << "\n";
}

// Hold a target at the sampling rate for a duration (start ramp / return).
void move_to(SmoothedArm& arm, const Joints& from, const Joints& to, double seconds, double rate,
             bool interruptible) {
    const auto period = std::chrono::duration<double>(1.0 / rate);
    const auto t0 = Clock::now();
    auto next = t0;
    while (!(interruptible && g_stop.load())) {
        const double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
        const double s = seconds > 1e-6 ? std::min(1.0, elapsed / seconds) : 1.0;
        Joints q{};
        for (int j = 0; j < kDof; ++j) q[j] = (1.0 - s) * from[j] + s * to[j];
        arm.set_target(q);
        if (elapsed >= seconds) break;
        next += std::chrono::duration_cast<Clock::duration>(period);
        std::this_thread::sleep_until(next);
    }
}

long collect(SmoothedArm& arm, const Options& o) {
    const Excitation& c = o.excitation;
    const bool is_static = c.profile == "static";
    std::ofstream csv(o.out);
    csv << "t,active_joint";
    for (const char* name : {"q_cmd_raw", "q_cmd_smooth", "q_real", "qd_real", "cur"})
        for (int j = 0; j < kDof; ++j) csv << "," << name << j;
    if (is_static) csv << ",pose_index,recording";
    csv << "\n";

    const auto period = std::chrono::duration<double>(1.0 / o.sample_hz);
    const double target_interval = 1.0 / o.set_target_hz;
    const auto t0 = Clock::now();
    auto next = t0;
    double last_target_t = -1e9;
    Joints raw = o.start_pose;
    long rows = 0;
    while (!g_stop.load()) {
        const double t = std::chrono::duration<double>(Clock::now() - t0).count();
        bool done = false, recording = true;
        int pose_index = -1;
        Joints target = o.start_pose;
        if (is_static) {
            target = static_target(c, o.start_pose, t, recording, done, pose_index);
        } else {
            target[c.joint] += excitation_offset(c, t, done);
        }
        if (done) break;
        // Zero-order hold at the policy rate, as on the robot.
        if (t - last_target_t >= target_interval) {
            raw = target;
            arm.set_target(raw);
            last_target_t = t;
        }

        const Joints smooth = arm.last_command();
        const Joints q = arm.position();
        const Joints qd = arm.velocity();
        const Joints current = arm.current();

        // Abort on a joint limit or on tracking error. SDK velocity is a scaled
        // motor-side value, so it is not used as a safety signal.
        std::string reason;
        for (int j = 0; j < kDof && reason.empty(); ++j) {
            if (q[j] < kJointLower[j] - 1e-3 || q[j] > kJointUpper[j] + 1e-3)
                reason = "joint " + std::to_string(j) + " outside its limit";
            else if (o.max_track_error > 0.0 && std::abs(q[j] - smooth[j]) > o.max_track_error)
                reason = "joint " + std::to_string(j) + " tracking error above threshold";
        }
        if (!reason.empty()) {
            std::fprintf(stderr, "[collect] abort at t=%.3f: %s\n", t, reason.c_str());
            arm.set_target(o.start_pose);
            break;
        }

        csv << t << "," << (is_static ? -1 : c.joint);
        for (const Joints* row : std::array<const Joints*, 5>{&raw, &smooth, &q, &qd, &current})
            for (int j = 0; j < kDof; ++j) csv << "," << (*row)[j];
        if (is_static) csv << "," << pose_index << "," << (recording ? 1 : 0);
        csv << "\n";
        ++rows;
        next += std::chrono::duration_cast<Clock::duration>(period);
        std::this_thread::sleep_until(next);
    }
    return rows;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    Excitation& c = o.excitation;
    std::string start_pose, static_poses;
    po::options_description desc("ARX R5 actuator-identification collector");
    desc.add_options()
        ("help,h", "show help")
        ("urdf", po::value(&o.urdf)->required(), "ARX SDK URDF (e.g. X5liteaa0.urdf)")
        ("can", po::value(&o.can)->required(), "CAN interface, e.g. can1")
        ("out", po::value(&o.out)->required(), "output CSV")
        ("profile", po::value(&c.profile)->default_value("chirp"), "chirp | multisine | step | static")
        ("joint", po::value(&c.joint)->default_value(1), "excited joint, zero-based 0..5")
        ("amp", po::value(&c.amplitude)->default_value(0.03), "excitation amplitude (rad)")
        ("f0", po::value(&c.f0)->default_value(0.1), "chirp start frequency (Hz)")
        ("f1", po::value(&c.f1)->default_value(3.0), "chirp end frequency (Hz)")
        ("dur", po::value(&c.duration)->default_value(30.0), "one-way sweep / multisine duration (s)")
        ("hold", po::value(&c.high_freq_hold)->default_value(2.0), "chirp high-frequency hold (s)")
        ("step-period", po::value(&c.step_period)->default_value(1.5), "step hold time (s)")
        ("step-count", po::value(&c.step_count)->default_value(20), "number of steps")
        ("step-seed", po::value(&c.step_seed)->default_value(0), "step amplitude seed")
        ("static-poses", po::value(&static_poses), "static: poses separated by ';', 6 comma values each")
        ("static-move", po::value(&c.static_move)->default_value(3.0), "static: move time per pose (s)")
        ("static-settle", po::value(&c.static_settle)->default_value(3.0), "static: unrecorded settle (s)")
        ("static-record", po::value(&c.static_record)->default_value(3.0), "static: recorded hold (s)")
        ("start-pose", po::value(&start_pose), "6 comma values (rad), default 0,1,0.8,0,0,0")
        ("ema-alpha", po::value(&o.ema_alpha)->default_value(0.05), "1 kHz EMA alpha; match deployment")
        ("sample-hz", po::value(&o.sample_hz)->default_value(100.0), "logging rate")
        ("set-target-hz", po::value(&o.set_target_hz)->default_value(50.0), "target rate (policy rate)")
        ("ramp-s", po::value(&o.ramp_s)->default_value(2.0), "ramp to the start pose (s)")
        ("return-s", po::value(&o.return_s)->default_value(1.0), "hold the start pose before exit (s)")
        ("max-track-err", po::value(&o.max_track_error)->default_value(0.35),
            "abort when |q - smooth command| exceeds this (rad); 0 disables");
    try {
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { std::cout << desc << "\n"; return 0; }
        po::notify(vm);
        if (!start_pose.empty()) o.start_pose = parse_joints(start_pose);
        std::stringstream poses(static_poses);
        std::string pose;
        while (std::getline(poses, pose, ';'))
            if (!pose.empty()) c.static_poses.push_back(parse_joints(pose));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "argument error: %s\n", e.what());
        return 2;
    }
    if (const std::string error = validate(o); !error.empty()) {
        std::fprintf(stderr, "invalid options: %s\n", error.c_str());
        return 2;
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::fprintf(stderr, "[collect] ema-alpha=%.4f (tau %.1f ms at 1 kHz); must match deployment\n",
                 o.ema_alpha, 1000.0 * 0.001 / -std::log(1.0 - o.ema_alpha));
    SmoothedArm arm(o.urdf, o.can, o.ema_alpha);

    std::atomic<bool> smoothing{true};
    std::thread ema([&]() {
        auto next = Clock::now();
        while (smoothing.load()) {
            arm.step_smooth();
            next += std::chrono::milliseconds(1);
            std::this_thread::sleep_until(next);
        }
    });

    move_to(arm, arm.position(), o.start_pose, o.ramp_s, o.sample_hz, true);
    long rows = 0;
    if (!g_stop.load()) {
        std::fprintf(stderr, "[collect] profile=%s\n", c.profile.c_str());
        rows = collect(arm, o);
        write_meta(o, rows);
    }
    // Return to the start pose even after Ctrl-C, then let the SDK hold it.
    const Joints here = arm.last_command();
    const double distance = [&] {
        double d = 0.0;
        for (int j = 0; j < kDof; ++j) d = std::max(d, std::abs(here[j] - o.start_pose[j]));
        return d;
    }();
    move_to(arm, here, o.start_pose, std::max(o.return_s, distance / kMaxStaticMoveSpeed), o.sample_hz, false);
    smoothing.store(false);
    ema.join();
    std::fprintf(stderr, "[collect] %ld rows -> %s\n", rows, o.out.c_str());
    return 0;
}
