from appium import webdriver
import time

# 连接移动设备所必须的参数
desired_caps = {}

# 当前要测试的设备的名称
desired_caps["deviceName"] = "127.0.0.1:62001"
# 系统
desired_caps["platformName"] = "Android"
# 系统的版本
desired_caps["platformVersion"] = "7.1.2"
# 要启动app的名称(包名)
desired_caps["appPackage"] = "com.ss.android.article.lite"
# 要启动的app的哪个界面
desired_caps["appActivity"] = ".activity.SplashActivity"

# 连接到Appium服务器
driver = webdriver.Remote(command_executor="http://127.0.0.1:4723/wd/hub", desired_capabilities=desired_caps)

# 等待应用启动
time.sleep(10)

# 点击第一个视频
first_video = driver.find_element_by_id("com.ss.android.article.lite:id/azu")
first_video.click()

# 等待视频加载
time.sleep(10)

# 在这里可以添加其他操作，例如观看视频的一些操作

# 打印当前页面源码
print(driver.page_source)

# 关闭app
driver.close_app()
# 释放资源
driver.quit()
