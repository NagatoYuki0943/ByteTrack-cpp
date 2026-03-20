#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <filesystem>
#include <cassert>
#include <nlohmann/json.hpp>
#include "src/ByteTrack/BYTETracker.h"

namespace fs = std::filesystem;

struct DetectionResult
{
    int frame_id;
    float prob;
    float x;
    float y;
    float width;
    float height;
};

struct TrackingResult
{
    int frame_id;
    int track_id;
    float x;
    float y;
    float width;
    float height;
};

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

int main(int argc, char *argv[])
{
    std::cout << "argc: " << argc << std::endl;
    std::cout << "program name: " << argv[0] << std::endl;

    const std::string detection_result_file = "../../../data/YOLOX_ncnn_palace/detection_results.json";
    const std::string tracking_result_file = "../../../data/YOLOX_ncnn_palace/tracking_results.json";
    if(!fs::exists(detection_result_file))
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
    std::map<int, std::vector<byte_track::Object>> detection_results;
    for(const auto &results_j : detection_j["results"]){
        int frame_id = std::stoi(results_j["frame_id"].get<std::string>());
        float prob = std::stof(results_j["prob"].get<std::string>());
        float x = std::stof(results_j["x"].get<std::string>());
        float y = std::stof(results_j["y"].get<std::string>());
        float width = std::stof(results_j["width"].get<std::string>());
        float height = std::stof(results_j["height"].get<std::string>());

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
    std::map<int, std::map<int, byte_track::Rect<float>>> tracking_results;
    for(const auto &results_j : tracking_j["results"]){
        int frame_id = std::stoi(results_j["frame_id"].get<std::string>());
        int track_id = std::stoi(results_j["track_id"].get<std::string>());
        float x = std::stof(results_j["x"].get<std::string>());
        float y = std::stof(results_j["y"].get<std::string>());
        float width = std::stof(results_j["width"].get<std::string>());
        float height = std::stof(results_j["height"].get<std::string>());

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
    for (const auto &[frame_id, objects] : detection_results)
    {
        auto &ref_objects = tracking_results[frame_id];
        const auto outputs = tracker.update(objects);

        std::cout << "frame_id: " << frame_id << std::endl;

        int predict_num = outputs.size();
        int ref_num = ref_objects.size();
        EXPECT_EQ(predict_num, ref_num);
        std::cout << "predict_num: " << predict_num << ", ref_num: " << ref_num << std::endl;

        for (const auto &outputs_per_frame : outputs)
        {
            const auto &rect = outputs_per_frame->getRect();
            const auto &track_id = outputs_per_frame->getTrackId();
            const auto &ref = ref_objects[track_id];
            EXPECT_NEAR(ref.x(), rect.x());
            EXPECT_NEAR(ref.y(), rect.y());
            EXPECT_NEAR(ref.width(), rect.width());
            EXPECT_NEAR(ref.height(), rect.height());

            std::cout << "track_id: " << track_id << ", x: " << rect.x() << ", y: " << rect.y() << ", width: " << rect.width() << ", height: " << rect.height() << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}
