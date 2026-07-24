/**
 * @file mpc_example.cpp
 * @brief Example of using `pjrt_exec` and `jax2exec` to call mpc jax from C++.
 */

#include <filesystem>
#include <iostream>

#include "third_party/call_jax_from_cpp/src/pjrt_exec/pjrt_exec.hpp"
#include "third_party/highfive/include/highfive/highfive.hpp"

int main() {
  // example setup
  const std::size_t num_samples = 2000;
  const std::string base_name = "./artifacts/mpc_export";

  const std::vector<double> earth_vstate0 = {
      0.00000000e+00,  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
      -3.69499823e-13, 3.78253326e+01, 0.00000000e+00, 0.00000000e+00,
      0.00000000e+00,  0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
      0.00000000e+00,  0.00000000e+00, 0.00000000e+00};
  // const std::vector<double> moon_vstate0 = {
  //     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
  //     3.53776427e-13, 6.26566416e+00, 0.00000000e+00, 0.00000000e+00,
  //     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
  //     0.00000000e+00, 0.00000000e+00, 0.00000000e+00};

  // mpc setup
  const std::vector<double> acc_ref = {1.0, 0.0, 9.81};
  const std::vector<double> omega_ref = {0.0, 0.0, 0.1};
  std::vector<double> prefilt0 = {
      0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
      0.00000000e+00, 0.00000000e+00, 1.90316721e-15, -1.19151129e-17,
      4.03144180e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
      0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
      0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00,
      0.00000000e+00};
  std::vector<double> filt0 = {0.0,        0.0, 0.00795775, 0.0,
                               0.07957747, 0.0, 0.0,        0.0};
  std::vector<double> vstate0_irl = earth_vstate0;
  std::vector<double> vstate0_sim = earth_vstate0;
  std::vector<double> xyz_hist = {0.0, 0.0, 0.1, 0.0, 0.0, 0.1};
  std::vector<double> yaw_hist = {0.0, 0.0};
  std::vector<double> quat_hist = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
  std::vector<double> last_control(200 * 6, 0.0);

  // results
  std::vector<double> y_xyz = {0.0, 0.0, 0.1};
  std::vector<double> y_yaw = {0.0};
  std::vector<double> y_tilt = {1.0, 0.0, 0.0};
  std::vector<std::vector<double>> y_xyz_res;
  std::vector<std::vector<double>> y_yaw_res;
  std::vector<std::vector<double>> y_tilt_res;
  std::vector<std::vector<double>> control_res;

  // pjrt setup
  auto client = std::make_shared<pjrt::Client>();
  auto devices = client->get_devices();
  auto device = devices[0];
  pjrt::AOTComputation aot_comp(base_name, client);

  // pjrt resuse
  std::shared_ptr<pjrt::Buffer> y_xyz_buff = pjrt::Buffer::to_device_blocking(
      y_xyz.data(), y_xyz.size(), client, device);
  std::shared_ptr<pjrt::Buffer> y_yaw_buff = pjrt::Buffer::to_device_blocking(
      y_yaw.data(), y_yaw.size(), client, device);
  std::shared_ptr<pjrt::Buffer> y_tilt_buff = pjrt::Buffer::to_device_blocking(
      y_tilt.data(), y_tilt.size(), client, device);
  std::shared_ptr<pjrt::Buffer> acc_ref_buff = pjrt::Buffer::to_device_blocking(
      acc_ref.data(), acc_ref.size(), client, device);
  std::shared_ptr<pjrt::Buffer> omega_ref_buff =
      pjrt::Buffer::to_device_blocking(omega_ref.data(), omega_ref.size(),
                                       client, device);
  std::shared_ptr<pjrt::Buffer> prefilt0_buff =
      pjrt::Buffer::to_device_blocking(prefilt0.data(), prefilt0.size(), client,
                                       device);
  std::shared_ptr<pjrt::Buffer> filt0_buff = pjrt::Buffer::to_device_blocking(
      filt0.data(), filt0.size(), client, device);
  std::shared_ptr<pjrt::Buffer> vstate0_irl_buff =
      pjrt::Buffer::to_device_blocking(vstate0_irl.data(), vstate0_irl.size(),
                                       client, device);
  std::shared_ptr<pjrt::Buffer> vstate0_sim_buff =
      pjrt::Buffer::to_device_blocking(vstate0_sim.data(), vstate0_sim.size(),
                                       client, device);
  std::shared_ptr<pjrt::Buffer> xyz_hist_buff =
      pjrt::Buffer::to_device_blocking(xyz_hist.data(), xyz_hist.size(), client,
                                       device);
  std::shared_ptr<pjrt::Buffer> yaw_hist_buff =
      pjrt::Buffer::to_device_blocking(yaw_hist.data(), yaw_hist.size(), client,
                                       device);
  std::shared_ptr<pjrt::Buffer> quat_hist_buff =
      pjrt::Buffer::to_device_blocking(quat_hist.data(), quat_hist.size(),
                                       client, device);
  std::shared_ptr<pjrt::Buffer> last_control_buff =
      pjrt::Buffer::to_device_blocking(last_control.data(), last_control.size(),
                                       client, device);
  std::vector<std::shared_ptr<pjrt::Buffer>> input_buffers = {
      acc_ref_buff,     omega_ref_buff,   prefilt0_buff, filt0_buff,
      vstate0_irl_buff, vstate0_sim_buff, xyz_hist_buff, yaw_hist_buff,
      quat_hist_buff,   last_control_buff};

  // random input timing
  std::vector<double> timings(num_samples);
  for (std::size_t i = 0; i < num_samples; ++i) {
    // start timing
    auto start = std::chrono::high_resolution_clock::now();

    // compute
    // note: in general, need to assign new acc_ref and omega_ref
    auto output_buffers = aot_comp.execute_blocking(input_buffers);
    for (std::size_t i = 2; i < input_buffers.size(); i++) {
      input_buffers[i] = output_buffers[i + 1];
    }
    output_buffers[0]->to_host_blocking(y_xyz.data(), y_xyz.size());
    output_buffers[1]->to_host_blocking(y_yaw.data(), y_yaw.size());
    output_buffers[2]->to_host_blocking(y_tilt.data(), y_tilt.size());
    output_buffers[10]->to_host_blocking(last_control.data(),
                                         last_control.size());

    // end timing
    auto end = std::chrono::high_resolution_clock::now();
    timings[i] =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    // update history
    y_xyz_res.push_back(y_xyz);
    y_yaw_res.push_back(y_yaw);
    y_tilt_res.push_back(y_tilt);
    control_res.push_back(last_control);
  }

  // compute average timing
  double avg_timing = 0.0;
  for (std::size_t i = 0; i < num_samples; ++i) {
    avg_timing += timings[i];
  }
  avg_timing /= num_samples;
  std::cout << "Average timing: " << avg_timing << " microseconds" << std::endl;

  // compute stddev timing
  double stddev_timing = 0.0;
  for (std::size_t i = 0; i < num_samples; ++i) {
    stddev_timing += (timings[i] - avg_timing) * (timings[i] - avg_timing);
  }
  stddev_timing = std::sqrt(stddev_timing / num_samples);
  std::cout << "Stddev timing: " << stddev_timing << " microseconds"
            << std::endl;

  // compute min and max timing
  std::size_t min_index = 1;
  std::size_t max_index = 1;
  for (std::size_t i = 2; i < num_samples; ++i) {
    if (timings[i] < timings[min_index]) {
      min_index = i;
    }
    if (timings[i] > timings[max_index]) {
      max_index = i;
    }
  }
  std::cout << "Min timing: " << timings[min_index] << " microseconds\n";
  std::cout << "Max timing: " << timings[max_index] << " microseconds\n";
  std::cout << "Min timing index: " << min_index << std::endl;
  std::cout << "Max timing index: " << max_index << std::endl;

  // not setinels?
  std::cout << "Output data: " << last_control[0] << ", " << last_control[1]
            << std::endl;

  // save interesting data to file (and use scoping for implicit file closing)
  {
    std::filesystem::create_directories("./data");
    HighFive::File file("./data/mpc_example_data.h5", HighFive::File::Truncate);
    file.createDataSet("y_xyz_res", y_xyz_res);
    file.createDataSet("y_yaw_res", y_yaw_res);
    file.createDataSet("y_tilt_res", y_tilt_res);
    file.createDataSet("control_res", control_res);
    file.createDataSet("timings", timings);
  }

  return 0;
}
