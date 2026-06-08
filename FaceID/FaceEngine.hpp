#pragma once
#include <vector>
#include <string>
#include <mutex>

// 纯 C++ 人脸识别引擎:存「姓名 + 向量」,做 L2 归一化 + 余弦比对 + 持久化。
// 不依赖任何平台 API,iOS 和安卓(NDK)都能用同一份。
class FaceEngine {
public:
    struct Match {
        std::string name;
        float score;   // 余弦相似度;库为空时 score = -2
    };

    explicit FaceEngine(const std::string& dbPath);

    void enroll(const std::string& name, const std::vector<float>& embedding);
    Match findBest(const std::vector<float>& embedding) const;
    int count() const;
    void clear();

private:
    struct Entry {
        std::string name;
        std::vector<float> emb;   // 已 L2 归一化
    };

    std::vector<Entry> entries_;
    std::string dbPath_;
    mutable std::mutex mu_;   // 后台检测线程读 / 主线程录入写,加锁防竞争

    void load();
    void save() const;
    static std::vector<float> l2normalize(const std::vector<float>& v);
    static float cosine(const std::vector<float>& a, const std::vector<float>& b);
};
