// face_cli — a tiny command-line front-end for the SAME pure-C++ FaceEngine that
// the iOS app uses (../../FaceID/FaceEngine.cpp). It exists to prove the engine is
// genuinely platform-independent: no Apple frameworks, builds + runs on any desktop.
//
//   make -C engine cli
//   ./engine/face_cli faces.db demo            # self-contained random demo
//   ./engine/face_cli faces.db enroll alice 0.1,0.2,0.3
//   ./engine/face_cli faces.db query 0.1,0.2,0.3
//   ./engine/face_cli faces.db list
//   ./engine/face_cli faces.db remove alice
//   ./engine/face_cli faces.db rename alice alicia
//
#include "FaceEngine.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sstream>

static std::vector<float> parseVec(const std::string& s) {
    std::vector<float> v; std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stof(tok));
    return v;
}

// deterministic pseudo-random unit-ish vector (no <random> dependency, seed-stable)
static std::vector<float> fakeVec(unsigned seed, int dim = 64) {
    std::vector<float> v(dim);
    unsigned s = seed * 2654435761u + 1013904223u;
    for (int i = 0; i < dim; ++i) { s = s * 1664525u + 1013904223u; v[i] = ((s >> 9) & 0xFFFF) / 65535.0f - 0.5f; }
    return v;
}

static void usage() {
    std::puts("usage: face_cli <db> {demo | enroll <name> <v,..> | query <v,..> | list | remove <name> | rename <old> <new>}");
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(); return 2; }
    std::string db = argv[1], cmd = argv[2];
    FaceEngine engine(db);

    if (cmd == "demo") {
        std::puts("[demo] enrolling 3 identities (2 templates each) with random vectors...");
        const char* who[] = {"alice", "bob", "carol"};
        for (int p = 0; p < 3; ++p) {
            engine.enroll(who[p], fakeVec(100 + p));
            engine.enroll(who[p], fakeVec(100 + p));   // a 2nd "shot" near the first
        }
        std::printf("[demo] persons=%d\n", engine.count());
        for (int p = 0; p < 3; ++p) {
            auto m = engine.findBest(fakeVec(100 + p));   // query == one of bob's templates
            std::printf("  query(%-6s) -> %-6s  cos=%.3f\n", who[p], m.name.c_str(), m.score);
        }
        auto stranger = engine.findBest(fakeVec(999));
        std::printf("  query(strangr) -> %-6s  cos=%.3f  (should be low)\n", stranger.name.c_str(), stranger.score);
        return 0;
    }
    if (cmd == "enroll" && argc == 5) { engine.enroll(argv[3], parseVec(argv[4]));
        std::printf("enrolled %s (now %d templates)\n", argv[3], engine.templateCount(argv[3])); return 0; }
    if (cmd == "query" && argc == 4) { auto m = engine.findBest(parseVec(argv[3]));
        std::printf("%s\t%.4f\n", m.name.c_str(), m.score); return 0; }
    if (cmd == "list") { std::printf("%d person(s):\n", engine.count());
        for (auto& n : engine.names()) std::printf("  %s (%d templates)\n", n.c_str(), engine.templateCount(n)); return 0; }
    if (cmd == "remove" && argc == 4) { std::printf(engine.remove(argv[3]) ? "removed %s\n" : "not found: %s\n", argv[3]); return 0; }
    if (cmd == "rename" && argc == 5) { std::printf(engine.rename(argv[3], argv[4]) ? "renamed\n" : "not found\n"); return 0; }

    usage(); return 2;
}
