# OptiSense Beta可编辑源码与固件

这是当前工作区的完整源码快照，不是旧Git提交导出；版本与逐文件SHA-256以根目录`MANIFEST.sha256.json`为准。包含最新Beta客户端源码、STM32F205RE单次ADC Beta v1.0.118源码及已验证固件包。

## 包的用途

本包用于受信任开发者修改代码，不是客户更新助手。客户使用单独交付的离线更新包；本包不包含运行客户端所需的Python解释器或客户安装器。历史兼容固件位于`outputs/ota`与`Python/firmware/usb`，不是推荐降级版本。

最新可烧录固件为`firmware/JDSU_F205RE_v1.0.118_beta.fbgfw`，映像CRC32为`1A951B61`。该映像已在三/四号机安装验证；四号机仍缺专属2001点校准表，不能套用另一台机器的表。仅含应用固件，不含私有完整Flash、Bootloader配置或现场密码。

## 开发环境与启动

Windows10/11 x64、CPython3.12 x64。在解压根目录执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_dev.ps1
.\start_demo.ps1
```

先在不连接设备的演示模式检查界面。源码Beta入口为`Python/launch_fbg_studio_beta.pyw`，使用`setup_dev.ps1`实际创建的虚拟环境解释器启动。旧`start_dev.ps1`不自动等于Beta入口；不要依据历史COM口或IP烧录、开光或驱动机器。

现行Python/JDSU/deployment/docs源码、PyInstaller配置、必要HAL/CMSIS库、登记校准表、测试夹具与已审查的出厂种子数据都保留。`Python/outputs`内只含从deployment种子生成的固定初始资源，未收录原工作站现场采集记录。

## 构建与验证

固件源码的单次ADC Beta分支为v1.0.118；v1.0.119轮询DAC分支未采纳，不能作为更高版本发布。构建前按`JDSU/AGENTS.md`运行温控、功能配置和扫描调度回归，统一40℃温控、USB更新与LAN采集要求不能回退。

```powershell
Push-Location Python
& <虚拟环境的python.exe> -m unittest test_thermal_fan_policy test_firmware_feature_profile test_firmware_scan_scheduler
Pop-Location
.\deployment\build_beta_firmware.ps1 -OutputName fw118_developer_rebuild
```

编译需要合法安装Keil MDK-ARM5/ARM Compiler6；脚本中的Keil路径需在新电脑核实。重编译成功不等于已烧录或光学精度验收通过。最新版发布说明见根README及docs验证记录。

原发行签名私钥不随包，`Python/update_public_key.py`只有公钥。开发者可以编辑和测试，但原客户认可的签名更新仍需持钥发布者签名，不应擅自替换信任公钥。

## 校验与再导出

```powershell
& <虚拟环境的python.exe> .\deployment\package_beta_source.py --verify "源码压缩包.zip"
& <虚拟环境的python.exe> .\deployment\package_beta_source.py --client-version <当前三段数字版本>
```

打包不连接硬件、不改变Git索引、不读取私钥、不覆盖旧ZIP。验证包含ZIP CRC、所有文件SHA-256、Python语法、登记的校准资源和完整固件头/映像CRC/SHA-256。未包含Git历史、缓存、venv、现场outputs/beta_data、大型训练模型和客户EXE。
