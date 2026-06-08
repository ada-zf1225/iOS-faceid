#include "FaceEngine.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

FaceEngine::FaceEngine(const std::string& dbPath) : dbPath_(dbPath) {
    load();
}

std::vector<float> FaceEngine::l2normalize(const std::vector<float>& v) {
    double sum = 0.0;
    for (float x : v) sum += static_cast<double>(x) * x;
    float norm = static_cast<float>(std::sqrt(sum));
    if (norm < 1e-10f) norm = 1e-10f;
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = v[i] / norm;
    return out;
}

float FaceEngine::cosine(const std::vector<float>& a, const std::vector<float>& b) {
    // a、b 已 L2 归一化 → 点积即余弦
    size_t n = std::min(a.size(), b.size());
    float dot = 0.f;
    for (size_t i = 0; i < n; ++i) dot += a[i] * b[i];
    return dot;
}

void FaceEngine::enroll(const std::string& name, const std::vector<float>& embedding) {
    std::lock_guard<std::mutex> lk(mu_);
    entries_.push_back({name, l2normalize(embedding)});
    save();
}

FaceEngine::Match FaceEngine::findBest(const std::vector<float>& embedding) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (entries_.empty()) return {"", -2.f};
    std::vector<float> q = l2normalize(embedding);
    const Entry* best = nullptr;
    float bestScore = -2.f;
    for (const auto& e : entries_) {
        float s = cosine(q, e.emb);     // 多模板:取所有模板里的最高分
        if (s > bestScore) { bestScore = s; best = &e; }
    }
    return {best->name, bestScore};
}

int FaceEngine::count() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::string> seen;
    for (const auto& e : entries_)
        if (std::find(seen.begin(), seen.end(), e.name) == seen.end())
            seen.push_back(e.name);
    return static_cast<int>(seen.size());
}

int FaceEngine::templateCount(const std::string& name) const {
    std::lock_guard<std::mutex> lk(mu_);
    int n = 0;
    for (const auto& e : entries_) if (e.name == name) ++n;
    return n;
}

std::vector<std::string> FaceEngine::names() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::string> seen;
    for (const auto& e : entries_)
        if (std::find(seen.begin(), seen.end(), e.name) == seen.end())
            seen.push_back(e.name);
    return seen;
}

bool FaceEngine::remove(const std::string& name) {
    std::lock_guard<std::mutex> lk(mu_);
    size_t before = entries_.size();
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                  [&](const Entry& e){ return e.name == name; }),
                   entries_.end());
    bool changed = entries_.size() != before;
    if (changed) save();
    return changed;
}

bool FaceEngine::rename(const std::string& oldName, const std::string& newName) {
    std::lock_guard<std::mutex> lk(mu_);
    bool changed = false;
    for (auto& e : entries_)
        if (e.name == oldName) { e.name = newName; changed = true; }
    if (changed) save();
    return changed;
}

void FaceEngine::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    entries_.clear();
    save();
}

void FaceEngine::save() const {
    std::ofstream f(dbPath_, std::ios::trunc);
    if (!f) return;
    for (const auto& e : entries_) {
        f << e.name << '\t';
        for (size_t i = 0; i < e.emb.size(); ++i) {
            if (i) f << ' ';
            f << e.emb[i];
        }
        f << '\n';
    }
}

void FaceEngine::load() {
    entries_.clear();
    std::ifstream f(dbPath_);
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        Entry e;
        e.name = line.substr(0, tab);
        std::istringstream ss(line.substr(tab + 1));
        float v;
        while (ss >> v) e.emb.push_back(v);
        if (!e.emb.empty()) entries_.push_back(std::move(e));
    }
}
