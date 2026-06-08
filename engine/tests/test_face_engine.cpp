// Unit tests for the pure-C++ FaceEngine — compiled against the SAME source the
// iOS app ships (../../FaceID/FaceEngine.cpp). No test framework: a tiny CHECK macro.
//
//   make -C engine test
//
#include "FaceEngine.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <thread>
#include <cmath>

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; std::printf("  FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); } \
} while (0)
#define CHECK_NEAR(a, b, eps, msg) CHECK(std::fabs((a) - (b)) < (eps), msg)

static std::string tmpdb(const char* tag) {
    std::string p = "/tmp/face_engine_test_";
    p += tag; p += ".db";
    std::remove(p.c_str());
    return p;
}
static std::vector<float> vec(std::initializer_list<float> xs) { return std::vector<float>(xs); }

int main() {
    // 1) empty engine
    {
        auto db = tmpdb("empty");
        FaceEngine e(db);
        CHECK(e.count() == 0, "empty count == 0");
        auto m = e.findBest(vec({1, 0, 0}));
        CHECK(m.score <= -1.f, "empty findBest score sentinel");
    }

    // 2) enroll + self-match ≈ 1, orthogonal ≈ 0
    {
        auto db = tmpdb("match");
        FaceEngine e(db);
        e.enroll("alice", vec({1, 0, 0, 0}));
        auto self = e.findBest(vec({2, 0, 0, 0}));      // same direction, different magnitude
        CHECK(self.name == "alice", "self match name");
        CHECK_NEAR(self.score, 1.0f, 1e-4f, "self match cosine ~1");
        auto orth = e.findBest(vec({0, 1, 0, 0}));
        CHECK_NEAR(orth.score, 0.0f, 1e-4f, "orthogonal cosine ~0");
    }

    // 3) multi-shot: a second template lifts recall for a pose the first misses
    {
        auto db = tmpdb("multishot");
        FaceEngine e(db);
        e.enroll("bob", vec({1, 0, 0, 0}));             // "frontal"
        std::vector<float> probe = {0.3f, 0.95f, 0, 0}; // "profile" — far from template 1
        float before = e.findBest(probe).score;
        e.enroll("bob", vec({0, 1, 0, 0}));             // add "profile" template
        float after = e.findBest(probe).score;
        CHECK(e.count() == 1, "multishot still one person");
        CHECK(e.templateCount("bob") == 2, "multishot two templates");
        CHECK(after > before + 0.3f, "multishot raises recall for hard probe");
        CHECK(e.findBest(probe).name == "bob", "multishot returns the person");
    }

    // 4) distinct count / templateCount / names order
    {
        auto db = tmpdb("count");
        FaceEngine e(db);
        e.enroll("a", vec({1, 0})); e.enroll("a", vec({0, 1})); e.enroll("b", vec({1, 1}));
        CHECK(e.count() == 2, "distinct persons == 2");
        CHECK(e.templateCount("a") == 2, "a has 2 templates");
        CHECK(e.templateCount("b") == 1, "b has 1 template");
        auto ns = e.names();
        CHECK(ns.size() == 2 && ns[0] == "a" && ns[1] == "b", "names first-seen order");
    }

    // 5) remove / rename
    {
        auto db = tmpdb("manage");
        FaceEngine e(db);
        e.enroll("x", vec({1, 0})); e.enroll("y", vec({0, 1}));
        CHECK(e.remove("x"), "remove existing returns true");
        CHECK(!e.remove("nope"), "remove missing returns false");
        CHECK(e.count() == 1, "count after remove");
        CHECK(e.rename("y", "z"), "rename returns true");
        auto ns = e.names();
        CHECK(ns.size() == 1 && ns[0] == "z", "rename applied");
    }

    // 6) persistence round-trip across instances
    {
        auto db = tmpdb("persist");
        { FaceEngine e(db); e.enroll("carol", vec({0.2f, 0.4f, 0.9f})); e.enroll("dave", vec({1, 0, 0})); }
        FaceEngine e2(db);                              // reload from disk
        CHECK(e2.count() == 2, "reload count");
        CHECK(e2.findBest(vec({0.2f, 0.4f, 0.9f})).name == "carol", "reload matches carol");
        CHECK(e2.templateCount("dave") == 1, "reload templateCount");
    }

    // 7) thread-safety smoke: concurrent enroll + query, no crash, consistent count
    {
        auto db = tmpdb("threads");
        FaceEngine e(db);
        std::thread w([&]{ for (int i = 0; i < 200; ++i) e.enroll("p" + std::to_string(i % 10), vec({(float)i, 1, 2})); });
        std::thread r([&]{ for (int i = 0; i < 200; ++i) { auto m = e.findBest(vec({(float)i, 1, 2})); (void)m; } });
        w.join(); r.join();
        CHECK(e.count() == 10, "threaded distinct count == 10");
    }

    std::printf("\n%s  %d passed, %d failed\n", g_fail == 0 ? "PASS" : "FAIL", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
