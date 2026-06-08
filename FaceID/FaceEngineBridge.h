#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// 比对结果(给 Swift 用的 ObjC 类型)
@interface FaceMatch : NSObject
@property (nonatomic, copy) NSString *name;
@property (nonatomic) float score;     // 余弦相似度
@end

/// C++ FaceEngine 的 Objective-C 包装。Swift 只见这个纯 ObjC 接口,
/// 真正的 C++ 藏在 .mm 实现里(Swift 不直接接触 C++,最稳)。
@interface FaceEngineBridge : NSObject
- (instancetype)initWithDBPath:(NSString *)path;
- (void)enrollName:(NSString *)name embedding:(NSArray<NSNumber *> *)embedding;
- (nullable FaceMatch *)findBest:(NSArray<NSNumber *> *)embedding;  // 库空返回 nil
- (NSInteger)count;
- (void)clear;
@end

NS_ASSUME_NONNULL_END
