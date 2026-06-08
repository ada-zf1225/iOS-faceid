#pragma once
#include <vector>
#include <string>
#include <mutex>

// 纯 C++ 人脸识别引擎:存「姓名 + 一个或多个向量模板」,做 L2 归一化 + 余弦比对 + 持久化。
// 不依赖任何平台 API,iOS 和安卓(NDK)/桌面 CLI 都能用同一份(见 ../engine/)。
//
// 多模板(multi-shot):对同一姓名多次 enroll 会各存一条模板;findBest 取「所有人所有模板」
// 里的最高余弦。多角度/戴不戴眼镜各录一张能显著提升召回。
class FaceEngine {
public:
    struct Match {
        std::string name;
        float score;   // 余弦相似度;库为空时 score = -2
    };

    explicit FaceEngine(const std::string& dbPath);

    // 录入一条模板(同名多次调用 = 多模板)。
    void enroll(const std::string& name, const std::vector<float>& embedding);
    // 最近邻:返回最高余弦所属姓名。
    Match findBest(const std::vector<float>& embedding) const;

    int count() const;                                  // 不同「人」的数量
    int templateCount(const std::string& name) const;   // 某人有几条模板
    std::vector<std::string> names() const;             // 去重姓名,按首次录入顺序
    bool remove(const std::string& name);               // 删除某人(其所有模板)
    bool rename(const std::string& oldName, const std::string& newName);  // 改名(合并到已存在的名)
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
    void save() const;        // 不加锁,调用方须已持有 mu_
    static std::vector<float> l2normalize(const std::vector<float>& v);
    static float cosine(const std::vector<float>& a, const std::vector<float>& b);
};
