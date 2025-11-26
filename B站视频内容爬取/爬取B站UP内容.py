import requests
import pandas as pd
import time
import random
import hashlib
import urllib.parse
import re
from functools import reduce
import os

# 尝试引入 AkShare（用于获取 A 股公司名单）
try:
    import akshare as ak
except ImportError:
    ak = None
    print("提示: 未安装 akshare，A 股公司名单将只使用内置热门公司列表。建议先运行: pip install akshare")


# ================= 用户配置区域 =================
UP_UID = '1274077132'  # 目标 UP 主 UID

# 你的 B 站 SESSDATA (注意保密)
SESSDATA = (
    'your_sessdata_here'
)
# 登陆 B 站后，F12 开发者模式从浏览器 Application-Cookie 里复制粘贴


# =============== 用 AkShare 获取 A 股公司名单 ===============

def load_company_db():
    """
    组合: AkShare A 股名单 + 手动维护的热门港美股/科技巨头
    不再使用东方财富接口。
    """
    companies = set()

    # 1. 用 AkShare 拿 A 股代码和简称
    if ak is not None:
        try:
            print("正在通过 AkShare 获取 A 股股票列表...")
            df = ak.stock_info_a_code_name()  # 获取沪深 A 股代码和简称
            # 兼容不同版本的列名: 优先找包含 name 或 abbr 的列
            name_cols = [c for c in df.columns if 'name' in c.lower() or 'abbr' in c.lower()]
            if name_cols:
                col = name_cols[0]
                a_names = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .replace('', pd.NA)
                    .dropna()
                    .tolist()
                )
                for n in a_names:
                    companies.add(n)
                    # 去掉 ST 标记
                    clean = n.replace('*ST', '').replace('ST', '').replace(' ', '')
                    if len(clean) >= 2:
                        companies.add(clean)
                print(f"AkShare 成功获取 A 股公司名称 {len(a_names)} 个，去重后共 {len(companies)} 个。")
            else:
                print("AkShare 返回结果中未找到公司名称列，将跳过 A 股列表。")
        except Exception as e:
            print(f"AkShare 获取 A 股列表失败，将仅使用手工名单。错误信息: {e}")
    else:
        print("未导入 akshare，将跳过 A 股列表。")

    # 2. 补充热门港美股/大厂 (A 股接口里没有这些)
    popular_companies = [
        "腾讯", "腾讯控股", "阿里巴巴", "阿里", "美团", "京东", "拼多多", "百度", "网易",
        "快手", "哔哩哔哩", "B站", "小米", "小米集团", "理想汽车", "蔚来", "小鹏汽车",
        "特斯拉", "苹果", "英伟达", "微软", "谷歌", "亚马逊", "台积电", "中芯国际",
        "宁德时代", "比亚迪", "药明康德", "东方甄选", "新东方", "好未来", "商汤",
        "瑞幸咖啡", "名创优品", "泡泡玛特", "知乎", "小红书", "字节跳动",
        "联发科", "联发科技"
    ]
    companies.update(popular_companies)

    total = len(companies)
    print(f"最终公司词库大小: {total}")

    # 按长度降序排列，优先匹配长名字（防止“腾讯控股”被匹配成“腾讯”）
    return sorted(list(companies), key=lambda x: len(x), reverse=True)


# 初始化加载词库 (程序启动时只运行一次)
KNOWN_COMPANIES = load_company_db()


def _looks_like_company(name: str) -> bool:
    """
    简单规则判断是否像一个公司名，用于兜底过滤噪音。
    只在 name 不在 KNOWN_COMPANIES 时才会用到。
    """
    name = name.strip()
    if len(name) < 2 or len(name) > 8:
        return False

    # 至少有两个中文字符
    cn = sum(1 for ch in name if '\u4e00' <= ch <= '\u9fff')
    if cn < 2:
        return False

    # 明确是行业 / 板块，而不是公司
    bad_exact = {
        "半导体", "新能源", "白酒", "军工", "地产", "银行", "券商",
        "煤炭", "钢铁", "医药", "消费", "指数", "板块", "行业", "股票"
    }
    if name in bad_exact:
        return False

    # 明显是问句/描述性质的词，不是名字
    bad_sub = [
        "是什么", "为何", "为什么", "能否", "是否",
        "护城河", "周期", "龙头", "板块", "生意",
        "估值", "涨跌", "回暖", "回调", "好不好", "贵不贵", "赚不赚钱"
    ]
    if any(w in name for w in bad_sub):
        return False

    # 一些固定的栏目 / 词语
    bad_words = {
        "公司大起底", "公司大", "大起底", "视频", "科普", "财经",
        "热点", "置顶", "回放", "全集", "完整版", "深度解析", "财报分析"
    }
    if name in bad_words:
        return False

    return True


def extract_company_name_smart(title: str) -> str:
    """
    智能提取公司名称：
    1）优先用词库（A 股 + 手工港美股）做“最长匹配”，可返回多个公司，用 '、' 连接
    2）再用规则：去掉栏目名前缀，匹配“XXX：”、“XXX，需要…”、“…—XXX”等模式
    3）实在不行再兜底返回“xxx(疑似)”或“需人工核对”
    """
    if not title:
        return "需人工核对"

    raw = title.strip()
    # 统一一些分隔符
    t = (raw
         .replace('丨', '|')
         .replace('｜', '|')
         .replace('——', '—')
         .replace('－', '—'))

    # -------- 1. 词库最长优先匹配（支持多个公司）--------
    results = []
    for name in KNOWN_COMPANIES:  # KNOWN_COMPANIES 已按长度降序
        if name and name in t:
            # 如果已经有一个更长的名字包含了它，就跳过（例如“阿里巴巴”已存在，则跳过“阿里”）
            if any(name in r for r in results):
                continue
            results.append(name)
    if results:
        return "、".join(results)

    # -------- 2. 去掉常见栏目名前缀，减少“公司大起底”干扰 --------
    # 例如：公司大起底丨新奥能源：...
    t = re.sub(r'^公司大起底[^|:：]*[|:：]', '', t).lstrip()

    noise_words = ["深扒", "起底", "揭秘", "聊聊", "复盘", "关于", "公司大起底"]
    for nw in noise_words:
        t = t.replace(nw, " ")

    def pick_candidate(candidate: str):
        """对候选字符串做清洗 + 词库 + 形态判断"""
        cand = candidate.strip()
        cand = re.sub(r'[，。,\.！？!？：:…\s]+$', '', cand)
        if not cand:
            return None

        # 先看词库里有没有
        if cand in KNOWN_COMPANIES:
            return cand

        # 尝试去掉常见后缀后再查词库
        suffixes = ["股份", "集团", "科技", "能源", "建材", "玻璃", "纸业", "动力", "银行", "生物", "电", "实业"]
        for suf in suffixes:
            if cand.endswith(suf):
                base = cand[:-len(suf)]
                if base in KNOWN_COMPANIES:
                    return base

        # 兜底：看形态像不像公司名
        if _looks_like_company(cand):
            return cand + "(疑似)"
        return None

    # -------- 3. VS 对比类：xxxVSyyyVSzzz --------
    if "VS" in t or "Vs" in t or "vs" in t:
        vs_clean = re.sub(r'(?i)vs', '|', t)
        parts = re.split(r'[|,，/、\s]+', vs_clean)
        multi = []
        for seg in parts:
            res = pick_candidate(seg)
            if res:
                base = res.replace("(疑似)", "")
                if base not in multi:
                    multi.append(base)
        if multi:
            return "、".join(multi)

    # -------- 4. “XXX：xxx” / “XXX: xxx” 结构，取冒号前最后一段 --------
    for colon in ['：', ':']:
        if colon in t:
            pre = t.split(colon)[0]
            parts = re.split(r'[|,，、\s]+', pre)
            for part in reversed(parts):   # 从右往左找，越靠右越可能是公司
                res = pick_candidate(part)
                if res:
                    return res
            break

    # -------- 5. “XXX，需要…” / “XXX，xxx” 开头 --------
    m = re.match(r'^([\u4e00-\u9fff]{2,10})[，,]', t)
    if m:
        res = pick_candidate(m.group(1))
        if res:
            return res

    # -------- 6. “XXX的……” 开头（信义玻璃的护城河是什么？）--------
    m = re.match(r'^([\u4e00-\u9fff]{2,10})的', t)
    if m:
        res = pick_candidate(m.group(1))
        if res:
            return res

    # -------- 7. “…的XXX” 结尾（自由现金流估值下的阿里巴巴）--------
    m = re.search(r'的([\u4e00-\u9fff]{2,10})[？\?]?', t)
    if m:
        res = pick_candidate(m.group(1))
        if res:
            return res

    # -------- 8. “XXX业绩/估值/经营效率为何…” 开头 --------
    m = re.match(r'^([\u4e00-\u9fff]{2,10}?)(?=(经营|未来|业绩|估值|是否|能否|周期|盈利|利润))', t)
    if m:
        res = pick_candidate(m.group(1))
        if res:
            return res

    # -------- 9. 再在整句里扫一遍“xxx股份 / xxx集团 / xxx科技 …” --------
    m = re.search(r'([\u4e00-\u9fff]{2,8}(股份|集团|科技|能源|纸业|建材|玻璃|生物|动力|银行|电|实业))', t)
    if m:
        res = pick_candidate(m.group(1))
        if res:
            return res

    # 实在识别不了
    return "需人工核对"


# ================= B 站 Wbi 签名算法 =================

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
        time.sleep(random.uniform(0.5, 1.5))
        url = 'https://api.bilibili.com/x/web-interface/nav'
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        json_content = resp.json()
        if json_content.get('code') != 0:
            print(f"获取密钥失败: code {json_content.get('code')}")
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
    print(f"正在初始化... (当前词库大小: {len(KNOWN_COMPANIES)})")
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
        sleep_time = random.uniform(3, 6)
        print(f"等待 {sleep_time:.1f} 秒后抓取第 {page} 页...")
        time.sleep(sleep_time)

        base_params = {
            'mid': mid,
            'ps': 30,
            'tid': 0,
            'pn': page,
            'keyword': '',
            'order': 'pubdate'
        }
        signed_params = enc_wbi(base_params, img_key, sub_key)

        try:
            url = 'https://api.bilibili.com/x/space/wbi/arc/search'
            response = requests.get(url, params=signed_params, headers=headers)

            if response.status_code != 200:
                print(f"HTTP 错误: {response.status_code}")
                break

            data = response.json()
            if data.get('code') != 0:
                print(f"API 错误: {data.get('message')}")
                break

            vlist = data.get('data', {}).get('list', {}).get('vlist', [])

            if not vlist:
                print("抓取完毕")
                break

            print(f"第 {page} 页获取到 {len(vlist)} 个视频，正在匹配公司名...")

            for v in vlist:
                title = v.get('title', '')
                company = extract_company_name_smart(title)

                videos.append({
                    '视频标题': title,
                    '涉及公司': company,
                    '发布时间': time.strftime('%Y-%m-%d', time.localtime(v.get('created', 0))),
                    '播放量': v.get('play', 0),
                    '链接': f"https://www.bilibili.com/video/{v.get('bvid', '')}"
                })

            page += 1

        except Exception as e:
            print(f"发生异常: {e}")
            break

    return pd.DataFrame(videos)


if __name__ == "__main__":
    if SESSDATA == 'your_sessdata_here':
        print("警告: 请先在代码中填入你的 SESSDATA")

    print(f"=== 启动全自动抓取 (UID: {UP_UID}) ===")
    df = get_all_videos(UP_UID, SESSDATA)

    if not df.empty:
        filename = f'B站数据_{UP_UID}_自动匹配版.xlsx'
        try:
            df.to_excel(filename, index=False)
            print(f"\n成功: 共获取 {len(df)} 条数据，已保存为: {filename}")
        except PermissionError:
            print(f"\n保存失败: 请关闭文件 '{filename}' 后重试")
    else:
        print("\n未获取到数据")