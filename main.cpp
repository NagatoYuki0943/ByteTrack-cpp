#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cassert>
#include <nlohmann/json.hpp>
#include "src/ByteTrack/BYTETracker.h"

namespace fs = std::filesystem;

nlohmann::json read_json(const std::string &json_file)
{
    nlohmann::json j;
    std::ifstream ifs(json_file);
    ifs >> j;
    ifs.close();
    return j;
}

double EPS = 1e-2;

void EXPECT_EQ(const int a, const int b)
{
    assert(a == b && "not equal");
}

void EXPECT_NEAR(const float a, const float b)
{
    assert((std::abs(a - b) < EPS) && "not near");
}

int test_byte_track()
{

    const std::string detection_result_file = "../../../data/YOLOX_ncnn_palace/detection_results.json";
    const std::string tracking_result_file = "../../../data/YOLOX_ncnn_palace/tracking_results.json";
    if (!fs::exists(detection_result_file))
    {
        std::cout << "detection results file: " << detection_result_file << " not found" << std::endl;
        return -1;
    }
    if (!fs::exists(tracking_result_file))
    {
        std::cout << "tracking results file: " << tracking_result_file << " not found" << std::endl;
        return -1;
    }

    // 读取检测结果
    auto detection_j = read_json(detection_result_file);
    int detection_fps = detection_j["fps"].get<int>();
    int detection_track_buffer = detection_j["track_buffer"].get<int>();
    std::map<size_t, std::vector<byte_track::Object>> detection_results;
    for (const auto &results_j : detection_j["results"])
    {
        int frame_id = std::stoi(results_j["frame_id"].get<std::string>());
        float prob = std::stof(results_j["prob"].get<std::string>());
        float x = std::stof(results_j["x"].get<std::string>());
        float y = std::stof(results_j["y"].get<std::string>());
        float width = std::stof(results_j["width"].get<std::string>());
        float height = std::stof(results_j["height"].get<std::string>());

        // tlwh(ltwh) format: Top-Left Width-Height
        byte_track::Object object = {byte_track::Rect(x, y, width, height), 0, prob};
        detection_results[frame_id].push_back(object);
    }
    std::cout << "detection fps: " << detection_fps << std::endl;
    std::cout << "detection track buffer: " << detection_track_buffer << std::endl;
    std::cout << "detection results size: " << detection_results.size() << std::endl;
    std::cout << std::endl;

    // 读取跟踪结果
    auto tracking_j = read_json(tracking_result_file);
    int tracking_fps = tracking_j["fps"].get<int>();
    int tracking_track_buffer = tracking_j["track_buffer"].get<int>();
    std::map<size_t, std::map<size_t, byte_track::Rect<float>>> tracking_results;
    for (const auto &results_j : tracking_j["results"])
    {
        int frame_id = std::stoi(results_j["frame_id"].get<std::string>());
        int track_id = std::stoi(results_j["track_id"].get<std::string>());
        float x = std::stof(results_j["x"].get<std::string>());
        float y = std::stof(results_j["y"].get<std::string>());
        float width = std::stof(results_j["width"].get<std::string>());
        float height = std::stof(results_j["height"].get<std::string>());

        // tlwh(ltwh) format: Top-Left Width-Height
        byte_track::Rect<float> rect = byte_track::Rect<float>(x, y, width, height);
        tracking_results[frame_id][track_id] = rect;
    }
    std::cout << "tracking fps: " << tracking_fps << std::endl;
    std::cout << "tracking track buffer: " << tracking_track_buffer << std::endl;
    std::cout << "tracking results size: " << tracking_results.size() << std::endl;
    std::cout << std::endl;

    // 创建跟踪器
    byte_track::BYTETracker tracker(detection_fps, detection_track_buffer);

    // 逐帧跟踪
    int time_total = 0;
    for (const auto &[frame_id, frame_objects] : detection_results)
    {
        auto &ref_track_outputs = tracking_results[frame_id];

        int t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        const auto track_outputs = tracker.update(frame_objects);
        int t2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        time_total += t2 - t1;

        std::cout << "frame_id: " << frame_id << std::endl;

        size_t track_num = track_outputs.size();
        size_t ref_track_num = ref_track_outputs.size();
        EXPECT_EQ(track_num, ref_track_num);
        std::cout << "track_num: " << track_num << ", ref_track_num: " << ref_track_num << std::endl;

        // 遍历每个 track 的 object
        for (const auto &track_output : track_outputs)
        {
            const auto &track_id = track_output->getTrackId();
            const auto &rect = track_output->getRect();

            const auto &ref = ref_track_outputs[track_id];

            EXPECT_NEAR(ref.x(), rect.x());
            EXPECT_NEAR(ref.y(), rect.y());
            EXPECT_NEAR(ref.width(), rect.width());
            EXPECT_NEAR(ref.height(), rect.height());

            std::cout << "track_id: " << track_id << ", x: " << rect.x() << ", y: " << rect.y() << ", width: " << rect.width() << ", height: " << rect.height() << std::endl;
        }
        std::cout << std::endl;
    }

    float avg_time = static_cast<float>(time_total) / tracking_results.size();
    std::cout << "avg_time: " << avg_time << std::endl;

    return 0;
}

int main(int argc, char *argv[])
{
    std::cout << "argc: " << argc << std::endl;
    std::cout << "program name: " << argv[0] << std::endl;

    int res = test_byte_track();
    if (res != 0)
        return -1;

    return 0;
}
