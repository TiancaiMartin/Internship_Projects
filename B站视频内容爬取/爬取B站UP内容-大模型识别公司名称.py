import requests
import pandas as pd
import time
import random
import hashlib
import urllib.parse
from functools import reduce
from openai import OpenAI  # 需要 pip install openai

# ================= 用户配置区域 =================
UP_UID = '1274077132'  # 目标 UP 主 UID
# 你的 B 站 SESSDATA (注意保密)
SESSDATA = (
    'your_sessdata_here'
)
# 登陆 B 站后，F12 开发者模式从浏览器 Application-Cookie 里复制粘贴

# DeepSeek API 配置
DEEPSEEK_API_KEY = 'sk-ac3456fa809c435fb60d79a41fc4a234'  # 替换为你自己的 Key
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


# ================= AI 智能识别模块 =================

def call_deepseek_api(title, summary):
    """
    调用 DeepSeek API，根据标题和简介提取公司名称
    """
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    # 构建提示词 (Prompt)
    # 这里的 Prompt 经过优化，强制要求只返回公司名，不废话
    prompt = f"""
    请分析以下B站财经视频的标题和简介，提取出视频主要讨论的【上市公司名称】。

    规则：
    1. 只返回公司名称（例如"腾讯控股"、"新奥能源"）。
    2. 如果涉及多个公司，用逗号分隔（例如"茅台,五粮液"）。
    3. 如果无法确定涉及具体公司，请返回"无"。
    4. 不要包含任何解释性文字，不要带书名号。

    视频标题：{title}
    视频简介：{summary}

    识别结果：
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # 低温度，让答案更确定
            stream=False,
        )
        result = response.choices[0].message.content.strip()
        # 清理一下可能残留的符号
        return result.replace("。", "").replace("识别结果：", "")
    except Exception as e:
        print(f"⚠️ AI 调用失败: {str(e)}")
        return "AI接口错误"


# ================= B站 Wbi 签名算法 (保持不变) =================
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

mixinKeyEncTab = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
    61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
    36, 20, 34, 44, 52
]


def get_mixin_key(orig: str):
    return reduce(lambda s, i: s + orig[i], mixinKeyEncTab, "")[:32]


def enc_wbi(params: dict, img_key: str, sub_key: str):
    mixin_key = get_mixin_key(img_key + sub_key)
    curr_time = round(time.time())
    params['wts'] = curr_time
    params = dict(sorted(params.items()))
    params = {
        k: ''.join(filter(lambda chr: chr not in "!'()*", str(v)))
        for k, v in params.items()
    }
    query = urllib.parse.urlencode(params)
    wbi_sign = hashlib.md5((query + mixin_key).encode()).hexdigest()
    params['w_rid'] = wbi_sign
    return params


def get_wbi_keys(sessdata):
    headers = {
        'User-Agent': USER_AGENT,
        'Cookie': f'SESSDATA={sessdata}'
    }
    try:
        # 稍微延迟一下
        time.sleep(random.uniform(0.5, 1.0))
        url = 'https://api.bilibili.com/x/web-interface/nav'
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        json_content = resp.json()
        if json_content.get('code') != 0:
            return None, None
        wbi_img = json_content['data']['wbi_img']
        img_key = wbi_img['img_url'].split('/')[-1].split('.')[0]
        sub_key = wbi_img['sub_url'].split('/')[-1].split('.')[0]
        return img_key, sub_key
    except Exception as e:
        print(f"获取密钥异常: {e}")
        return None, None


# ================= 爬虫主逻辑 =================

def get_all_videos(mid, sessdata):
    print(f"正在初始化... 准备使用 DeepSeek AI 进行智能识别")
    img_key, sub_key = get_wbi_keys(sessdata)
    if not img_key:
        print("密钥获取失败，请检查 SESSDATA")
        return pd.DataFrame()

    videos = []
    page = 1
    headers = {
        'User-Agent': USER_AGENT,
        'Cookie': f'SESSDATA={sessdata}',
        'Referer': f'https://space.bilibili.com/{mid}'
    }

    while True:
        # 适当延迟，防止B站封IP
        sleep_time = random.uniform(2, 4)
        print(f"等待 {sleep_time:.1f}s 后抓取第 {page} 页列表...")
        time.sleep(sleep_time)

        base_params = {'mid': mid, 'ps': 30, 'tid': 0, 'pn': page, 'keyword': '', 'order': 'pubdate'}
        signed_params = enc_wbi(base_params, img_key, sub_key)

        try:
            url = 'https://api.bilibili.com/x/space/wbi/arc/search'
            response = requests.get(url, params=signed_params, headers=headers)

            if response.status_code != 200:
                print(f"HTTP错误: {response.status_code}")
                break

            data = response.json()
            if data.get('code') != 0:
                print(f"API错误: {data.get('message')}")
                break

            vlist = data.get('data', {}).get('list', {}).get('vlist', [])

            if not vlist:
                print("所有页面抓取完毕！")
                break

            print(f"-> 第 {page} 页获取到 {len(vlist)} 个视频，开始 AI 识别...")

            # 遍历当前页面的视频
            for i, v in enumerate(vlist):
                title = v.get('title', '').strip()
                # 获取简介/摘要 (字段通常是 description 或 desc)
                summary = v.get('description', '').strip().replace('\n', ' ')

                # 如果简介太长，截取前200字省钱且够用
                if len(summary) > 300:
                    summary = summary[:300] + "..."

                # === 调用 AI 识别 ===
                # 打印进度，因为AI调用需要时间
                print(f"   [{i + 1}/{len(vlist)}] 正在分析: {title[:15]}...", end="", flush=True)

                company = call_deepseek_api(title, summary)

                print(f" -> 识别结果: 【{company}】")

                videos.append({
                    '视频标题': title,
                    '涉及公司': company,  # AI 识别结果
                    '视频摘要': summary,  # 新增摘要列
                    '发布时间': time.strftime('%Y-%m-%d', time.localtime(v.get('created', 0))),
                    '播放量': v.get('play', 0),
                    '链接': f"https://www.bilibili.com/video/{v.get('bvid', '')}"
                })

                # ⚠️ 每次调用 AI 后稍微停顿一下，防止触发 API 速率限制 (QPS)
                # 如果你是付费的高级账号可以去掉这个 sleep
                time.sleep(0.5)

            page += 1

        except Exception as e:
            print(f"发生异常: {e}")
            break

    return pd.DataFrame(videos)


if __name__ == "__main__":
    if SESSDATA == 'your_sessdata_here':
        print("【警告】请先在代码中填入你的 SESSDATA！")

    print(f"=== 启动 AI 增强版抓取 (UID: {UP_UID}) ===")
    df = get_all_videos(UP_UID, SESSDATA)

    if not df.empty:
        filename = f'B站数据_{UP_UID}_AI识别版.xlsx'
        try:
            df.to_excel(filename, index=False)
            print(f"\n✅ 成功！共获取 {len(df)} 条数据")
            print(f"📂 文件已保存为: {filename}")
        except PermissionError:
            print(f"\n❌ 保存失败！请关闭文件 '{filename}' 后重试")
    else:
        print("\n❌ 未获取到数据")