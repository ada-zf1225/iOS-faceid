# 怎么生成 README 顶部的 Demo GIF

README 顶部引用 `docs/demo.gif`。录一段真机演示放进来即可:

1. **录屏**:iPad 上跑 App,演示「录入一张脸 → 同人变绿框认出 → 换个人是红框陌生人 → 拖阈值滑块」。
   - 用 iOS 自带「屏幕录制」(控制中心),或 Mac 上 QuickTime「影片录制」选 iPad 当摄像头来源录。
2. **转 GIF + 压缩**(在 Mac 上,装了 ffmpeg):
   ```bash
   # 截取 ~8 秒、宽 280、10fps,体积控制在 ~3MB 内
   ffmpeg -i screen.mov -ss 0 -t 8 -vf "fps=10,scale=280:-1:flags=lanczos" -loop 0 docs/demo.gif
   ```
   或用 https://ezgif.com 在线转。
3. 把 `docs/demo.gif` 放好,README 顶部就会显示。

> 隐私提示:演示别录到无关人脸/敏感信息;公开仓库前确认 GIF 内容 OK。
