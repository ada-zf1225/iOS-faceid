#import "FaceEngineBridge.h"
#include "FaceEngine.hpp"
#include <vector>
#include <string>

@implementation FaceMatch
@end

@implementation FaceEngineBridge {
    FaceEngine *_engine;   // 持有 C++ 引擎
}

- (instancetype)initWithDBPath:(NSString *)path {
    if (self = [super init]) {
        _engine = new FaceEngine(std::string(path.UTF8String));
    }
    return self;
}

- (void)dealloc {
    delete _engine;
}

static std::vector<float> ToVec(NSArray<NSNumber *> *arr) {
    std::vector<float> v;
    v.reserve(arr.count);
    for (NSNumber *n in arr) v.push_back(n.floatValue);
    return v;
}

- (void)enrollName:(NSString *)name embedding:(NSArray<NSNumber *> *)embedding {
    _engine->enroll(std::string(name.UTF8String), ToVec(embedding));
}

- (nullable FaceMatch *)findBest:(NSArray<NSNumber *> *)embedding {
    FaceEngine::Match m = _engine->findBest(ToVec(embedding));
    if (m.score < -1.5f) return nil;   // 库为空的哨兵值
    FaceMatch *r = [FaceMatch new];
    r.name = [NSString stringWithUTF8String:m.name.c_str()];
    r.score = m.score;
    return r;
}

- (NSInteger)count { return _engine->count(); }

- (NSArray<NSString *> *)names {
    NSMutableArray<NSString *> *out = [NSMutableArray array];
    for (const auto &n : _engine->names())
        [out addObject:[NSString stringWithUTF8String:n.c_str()]];
    return out;
}

- (NSInteger)templateCountForName:(NSString *)name {
    return _engine->templateCount(std::string(name.UTF8String));
}

- (BOOL)removeName:(NSString *)name {
    return _engine->remove(std::string(name.UTF8String)) ? YES : NO;
}

- (BOOL)renameFrom:(NSString *)oldName to:(NSString *)newName {
    return _engine->rename(std::string(oldName.UTF8String), std::string(newName.UTF8String)) ? YES : NO;
}

- (void)clear { _engine->clear(); }

@end
