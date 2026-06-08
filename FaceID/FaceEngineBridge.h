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
- (void)enrollName:(NSString *)name embedding:(NSArray<NSNumber *> *)embedding;  // 同名多次=多模板
- (nullable FaceMatch *)findBest:(NSArray<NSNumber *> *)embedding;  // 库空返回 nil
- (NSInteger)count;                                  // 不同「人」数
- (NSArray<NSString *> *)names;                      // 去重姓名,按首次录入顺序
- (NSInteger)templateCountForName:(NSString *)name NS_SWIFT_NAME(templateCount(of:));  // 某人模板数
- (BOOL)removeName:(NSString *)name;                 // 删除某人
- (BOOL)renameFrom:(NSString *)oldName to:(NSString *)newName;
- (void)clear;
@end

NS_ASSUME_NONNULL_END
