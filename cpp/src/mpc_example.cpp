/**
 * @file mpc_example.cpp
 * @brief Example of using `pjrt_exec` and `jax2exec` to call mpc jax from C++.
 */

#include <filesystem>
#include <iostream>

#include "third_party/call_jax_from_cpp/src/pjrt_exec/pjrt_exec.hpp"
#include "third_party/highfive/include/highfive/highfive.hpp"

enum class PersonelMode {
  UNSUITED = 0,
  SPECIAL = 1,
  UNSUITED_LEFT = 2,
  UNSUITED_RIGHT = 3,
  NONE = 4,
  SUITED = 5,
};

enum class PredictionMode {
  CONSTANT = 0,
  LANDER = 1,
};

enum class WeightMode {
  CONSTANT = 0,
  LANDER_00 = 1,
  LANDER_01 = 2,
};

int main() {
  // read data
  std::vector<std::vector<double>> ref_data;
  const std::vector<std::string> ref_file_names = {"../data/cpp_const.hdf",
                                                   "../data/cpp_lander.hdf",
                                                   "../data/cpp_rover.hdf"};
  {
    HighFive::File file(ref_file_names[0], HighFive::File::ReadOnly);
    file.getDataSet("data/block0_values").read(ref_data);
  }
  std::vector<std::vector<double>> acc_ref(ref_data.size(),
                                           std::vector<double>(3, 0.0));
  std::vector<std::vector<double>> omega_ref(ref_data.size(),
                                             std::vector<double>(3, 0.0));
  for (std::size_t i = 0; i < ref_data.size(); i++) {
    acc_ref[i][0] = ref_data[i][0];
    acc_ref[i][1] = ref_data[i][1];
    acc_ref[i][2] = ref_data[i][2];
    omega_ref[i][0] = ref_data[i][3];
    omega_ref[i][1] = ref_data[i][4];
    omega_ref[i][2] = ref_data[i][5];
  }

  // initial vstate examples
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
  // const std::vector<double> lander_vstate0 = {
  //     0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
  //     0.0000000000000000e+00,
  //     -4.2010950661908945e-13, 1.6291420859845125e+01,
  //     0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
  //     0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
  //     0.0000000000000000e+00, 0.0000000000000000e+00,
  //     0.0000000000000000e+00};

  // mpc setup
  const std::vector<double> personel_mode = {(double)PersonelMode::UNSUITED};
  const std::vector<double> prediction_mode = {
      (double)PredictionMode::CONSTANT};
  const std::vector<double> weight_mode = {(double)WeightMode::CONSTANT};
  std::vector<double> last_control(200 * 6, 0.0);
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
  std::vector<double> y_vest_sim_hist(4 * 6, 0.0);
  std::vector<double> xyz_hist = {0.0, 0.0, 0.1, 0.0, 0.0, 0.1};
  std::vector<double> yaw_hist = {0.0, 0.0};
  std::vector<double> quat_hist = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
  std::vector<double> terminal_param = {0.0};
  std::vector<double> iter = {0};

  // results
  std::vector<double> u_xyz = {0.0, 0.0, 0.1};
  std::vector<double> u_yaw = {0.0};
  std::vector<double> u_tilt = {1.0, 0.0, 0.0};
  std::vector<std::vector<double>> u_xyz_res;
  std::vector<std::vector<double>> u_yaw_res;
  std::vector<std::vector<double>> u_tilt_res;
  std::vector<std::vector<double>> control_res;

  // pjrt setup
  auto client = std::make_shared<pjrt::Client>();
  auto devices = client->get_devices();
  auto device = devices[0];
  auto init_data = [client, device](std::vector<double> data) {
    return pjrt::Buffer::to_device_blocking(data.data(), data.size(), client,
                                            device);
  };
  const std::string base_name = "./artifacts/mpc_export";
  pjrt::AOTComputation aot_comp(base_name, client);

  // pjrt input resuse
  auto personel_mode_buff = init_data(personel_mode);
  auto prediction_mode_buff = init_data(prediction_mode);
  auto weight_mode_buff = init_data(weight_mode);
  auto acc_ref_buff = init_data(acc_ref[0]);
  auto omega_ref_buff = init_data(omega_ref[0]);
  auto prefilt0_buff = init_data(prefilt0);
  auto last_control_buff = init_data(last_control);
  auto filt0_buff = init_data(filt0);
  auto vstate0_irl_buff = init_data(vstate0_irl);
  auto vstate0_sim_buff = init_data(vstate0_sim);
  auto y_vest_sim_hist_buff = init_data(y_vest_sim_hist);
  auto xyz_hist_buff = init_data(xyz_hist);
  auto yaw_hist_buff = init_data(yaw_hist);
  auto quat_hist_buff = init_data(quat_hist);
  auto terminal_param_buff = init_data(terminal_param);
  auto iter_buff = init_data(iter);
  std::vector<std::shared_ptr<pjrt::Buffer>> input_buffers = {
      personel_mode_buff, prediction_mode_buff, weight_mode_buff,
      acc_ref_buff,       omega_ref_buff,       last_control_buff,
      prefilt0_buff,      filt0_buff,           vstate0_irl_buff,
      vstate0_sim_buff,   y_vest_sim_hist_buff, xyz_hist_buff,
      yaw_hist_buff,      quat_hist_buff,       terminal_param_buff,
      iter_buff,
  };

  // random input timing
  const std::size_t num_samples = ref_data.size();
  std::vector<double> timings(num_samples);
  for (std::size_t i = 0; i < num_samples; ++i) {
    // start timing
    auto start = std::chrono::high_resolution_clock::now();

    // compute
    input_buffers[3] = init_data(acc_ref[i]);
    input_buffers[4] = init_data(omega_ref[i]);
    auto output_buffers = aot_comp.execute_blocking(input_buffers);
    for (std::size_t i = 3; i < output_buffers.size(); i++) {
      input_buffers[i + 2] = output_buffers[i];
    }
    output_buffers[0]->to_host_blocking(u_xyz.data(), u_xyz.size());
    output_buffers[1]->to_host_blocking(u_yaw.data(), u_yaw.size());
    output_buffers[2]->to_host_blocking(u_tilt.data(), u_tilt.size());
    output_buffers[3]->to_host_blocking(last_control.data(),
                                        last_control.size());

    // end timing
    auto end = std::chrono::high_resolution_clock::now();
    timings[i] =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    // update history
    u_xyz_res.push_back(u_xyz);
    u_yaw_res.push_back(u_yaw);
    u_tilt_res.push_back(u_tilt);
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
    file.createDataSet("u_xyz_res", u_xyz_res);
    file.createDataSet("u_yaw_res", u_yaw_res);
    file.createDataSet("u_tilt_res", u_tilt_res);
    file.createDataSet("control_res", control_res);
    file.createDataSet("timings", timings);
  }

  return 0;
}
