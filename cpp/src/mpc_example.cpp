/**
 * @file mpc_example.cpp
 * @brief Example of using `pjrt_exec` and `jax2exec` to call mpc jax from C++.
 */

#include <filesystem>
#include <iostream>
#include <thread>

#include "third_party/call_jax_from_cpp/src/pjrt_exec/pjrt_exec.hpp"
#include "third_party/highfive/include/highfive/highfive.hpp"

int main() {
  // example setup
  const std::size_t num_samples = 4000;
  const std::string base_name = "./artifacts/mpc_export";

  // mpc setup
  const double dt = 0.005;  // 200 hz
  const std::vector<double> acc_ref = {1.0, 0.0, 9.81};
  const std::vector<double> omega_ref = {0.0, 0.0, 0.1};
  std::vector<double> rstate0(2 * 6, 0.0);
  std::vector<double> vstate0_irl(15, 0.0);
  std::vector<double> vstate0_sim(15, 0.0);
  std::vector<double> control0(6, 0.0);
  std::vector<double> last_control(200 * 6, 0.0);
  std::vector<std::vector<double>> control_hist;

  vstate0_irl[4] = 9.18440318e-06;  // earth
  vstate0_irl[5] = 3.78252396e01;   // earth
  vstate0_sim[4] = 9.18440318e-06;  // earth
  vstate0_sim[5] = 3.78252396e01;   // earth

  // pjrt setup
  auto client = std::make_shared<pjrt::Client>();
  auto devices = client->get_devices();
  auto device = devices[0];
  pjrt::AOTComputation aot_comp(base_name, client);

  // WARNING: sleep to avoid segfault
  // without sleeping, the BUFFER::to_device call sometimes segfaults
  // I do not now what the optimal sleep time is, but 1ms is sufficient
  std::this_thread::sleep_for(std::chrono::microseconds(1000));

  // random input timing
  std::vector<double> timings(num_samples);
  for (std::size_t i = 0; i < num_samples; ++i) {
    // start timing
    auto start = std::chrono::high_resolution_clock::now();

    // compute
    std::vector<std::shared_ptr<pjrt::Buffer>> input_buffers = {
        pjrt::Buffer::to_device_blocking(acc_ref.data(), acc_ref.size(), client,
                                         device),
        pjrt::Buffer::to_device_blocking(omega_ref.data(), omega_ref.size(),
                                         client, device),
        pjrt::Buffer::to_device_blocking(rstate0.data(), rstate0.size(), client,
                                         device),
        pjrt::Buffer::to_device_blocking(vstate0_irl.data(), vstate0_irl.size(),
                                         client, device),
        pjrt::Buffer::to_device_blocking(vstate0_sim.data(), vstate0_sim.size(),
                                         client, device),
        pjrt::Buffer::to_device_blocking(control0.data(), control0.size(),
                                         client, device),
        pjrt::Buffer::to_device_blocking(last_control.data(),
                                         last_control.size(), client, device),
    };
    auto output_buffers = aot_comp.execute_blocking(input_buffers);
    output_buffers[0]->to_host_blocking(last_control.data(),
                                        last_control.size());
    output_buffers[1]->to_host_blocking(vstate0_irl.data(), vstate0_irl.size());
    output_buffers[2]->to_host_blocking(vstate0_sim.data(), vstate0_sim.size());

    // end timing
    auto end = std::chrono::high_resolution_clock::now();
    timings[i] =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    // update history
    control_hist.push_back(last_control);
    for (std::size_t i = 0; i < 6; i++) {
      control0[i] = last_control[i];
      rstate0[i] =
          rstate0[i] + dt * rstate0[i + 6] + 0.5 * dt * dt * last_control[i];
      rstate0[i + 6] = rstate0[i + 6] + dt * last_control[i];
    }
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
    file.createDataSet("control_hist", control_hist);
    file.createDataSet("timings", timings);
  }

  return 0;
}
