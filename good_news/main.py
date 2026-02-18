
import feedparser
import google.generativeai as genai
import schedule
import time
import os
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)
# Use GEMINI_MODEL from env, default to gemini-3-flash-preview
model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
print(f"Using Gemini Model: {model_name}")
model = genai.GenerativeModel(model_name)


RSS_FEEDS = [
    "https://kr.investing.com/rss/news.rss",
    "https://www.yna.co.kr/rss/market.xml",
    "https://www.yna.co.kr/rss/economy.xml",
    "https://www.mk.co.kr/rss/50200011/",
    "https://www.mk.co.kr/rss/30100041/",
    "https://www.hankyung.com/feed/finance",
    "https://www.hankyung.com/feed/economy",
    "https://www.thevaluenews.co.kr/rss_view.php?code=m663w38",
    "https://www.thevaluenews.co.kr/rss_view.php?code=m71n1f6",
    "https://www.thevaluenews.co.kr/rss_view.php?code=m76i0t2",
    "https://www.thevaluenews.co.kr/rss_view.php?code=m65gpg7",
    "https://rss.etoday.co.kr/eto/market_news.xml",
    "https://rss.etoday.co.kr/eto/finance_news.xml",
    "https://rss.etoday.co.kr/eto/economy_news.xml"
]

seen_links = set()



def fetch_and_analyze():
    # Use system local time for log readability if preferred, but keeping explicit about what time it is is good.
    # Let's just print "뉴스 확인 중..."
    print(f"[{datetime.datetime.now()}] 뉴스 확인 중...")
    
    # Check news from the last 6 hours (increased for testing/startup)
    current_time = datetime.datetime.now(datetime.timezone.utc)
    time_threshold = current_time - datetime.timedelta(hours=6)
    
    found_any = False
    
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                link = entry.get('link', '')
                if not link or link in seen_links:
                    continue
                
                # Check publication time
                published_time = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    # feedparser parses to UTC struct_time
                    published_time = datetime.datetime(*entry.published_parsed[:6], tzinfo=datetime.timezone.utc)
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                     published_time = datetime.datetime(*entry.updated_parsed[:6], tzinfo=datetime.timezone.utc)
                
                if published_time:
                    if published_time < time_threshold:
                        continue
                else:
                    pass

                seen_links.add(link)
                found_any = True
                
                print(f"새로운 뉴스 발견: {entry.title}")
                analyze_news(entry)
                
        except Exception as e:
            print(f"RSS 피드 가져오기 오류 ({url}): {e}")
            
    if not found_any:
        print("관련된 새로운 뉴스가 없습니다.")

def analyze_news(entry):
    prompt = f"""
    Analyze the following stock market news article.
    
    Title: {entry.title}
    Link: {entry.link}
    Summary: {entry.get('summary', 'No summary provided')}
    
    Task:
    1. Summarize the key points of the news in Korean.
    2. Determine if this news is a "호재" (Good News), "악재" (Bad News), or "중립" (Neutral) for the stock market or specific companies.
    3. Identify the main companies mentioned.
    4. Identify related companies or sectors that might be affected.
    
    IMPORTANT: ALL OUTPUT MUST BE IN KOREAN.
    
    Output Format:
    --------------------------------------------------
    [뉴스 분석]
    제목: {entry.title}
    감정: <호재/악재/중립>
    요약: <한국어 요약>
    관련 기업: <기업 A, 기업 B...>
    연관 섹터/기업: <기업 X, 기업 Y...>
    원본 링크: {entry.link}
    --------------------------------------------------
    """
    
    try:
        response = model.generate_content(prompt)
        print(response.text)
    except Exception as e:
        print(f"Gemini 분석 오류: {e}")

def main():
    print("주식 뉴스 모니터를 시작합니다...")
    print("중지하려면 Ctrl+C를 누르세요.")
    
    # Run once immediately
    fetch_and_analyze()
    
    schedule.every(1).minutes.do(fetch_and_analyze)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()

