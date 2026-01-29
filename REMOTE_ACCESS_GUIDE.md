# 远程服务器运行 Web Demo 本地访问指南

本指南介绍如何在 Linux 服务器上运行 PG-MoE 的 Web Demo，并在 Windows 本地浏览器中进行操作。

## 前置准备

确保您已安装项目依赖：
```bash
cd /hy-tmp/PG-MOE-Project/code
pip install gradio
```

---

## 方法一：使用 Gradio 公网链接 (最简单，无需配置)

Gradio 自带内网穿透功能。我们的 `web_demo.py` 默认已开启 `share=True`。

1. **运行脚本**：
   ```bash
   python web_demo.py
   ```

2. **获取链接**：
   等待几秒钟，终端会输出类似以下内容：
   ```text
   Running on local URL:  http://0.0.0.0:7860
   Running on public URL: https://e2b3c4d5f6a7.gradio.live  <-- 复制这个链接
   ```

3. **访问**：
   直接在 Windows 浏览器中打开那个 `https://....gradio.live` 链接即可使用。
   *注意：该链接有效期通常为 72 小时。*

---

## 方法二：VS Code 端口转发 (推荐，稳定)

如果您是使用 VS Code 的 Remote - SSH 插件连接服务器，这是最稳定的方式。

1. **运行脚本**：
   在 VS Code 的集成终端中运行：
   ```bash
   python web_demo.py
   ```

2. **自动/手动转发**：
   *   **自动**：VS Code 通常会检测到 `7860` 端口被占用，并在右下角弹出提示 "Open in Browser"。直接点击即可。
   *   **手动**：
        1. 打开 VS Code 底部面板（终端所在区域），点击 **"PORTS" (端口)** 选项卡。
        2. 点击 **"Add Port" (添加端口)**。
        3. 输入端口号 `7860` 并回车。
        4. 您会看到 "Local Address" 列出现 `localhost:7860`。

3. **访问**：
   点击 VS Code 中的 `localhost:7860` 链接，或在 Windows 浏览器手动输入 `http://localhost:7860`。

---

## 方法三：SSH 命令行隧道 (传统方式)

如果您使用 PowerShell、CMD 或 Putty 连接服务器。

1. **建立 SSH 连接时映射端口**：
   在 Windows 终端中，使用 `-L` 参数修改您的 SSH 命令：
   
   ```powershell
   # 语法: ssh -L 本地端口:127.0.0.1:远程端口 用户名@IP -p SSH端口
   ssh -L 7860:127.0.0.1:7860 root@123.123.123.123 -p 22
   ```

2. **运行脚本**：
   连接成功后，在服务器运行：
   ```bash
   python web_demo.py
   ```

3. **访问**：
   在 Windows 浏览器访问 `http://127.0.0.1:7860`。

---

## 常见问题

**Q: 端口 7860 被占用了怎么办？**
A: Gradio 会自动顺延端口（如 7861, 7862）。请观察终端输出的 `Running on local URL` 中的端口号，并相应调整上述步骤中的端口设置。
