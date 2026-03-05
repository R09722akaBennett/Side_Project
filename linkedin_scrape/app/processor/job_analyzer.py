import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime, timedelta
import time
from openai import OpenAI
from sqlalchemy import text

from app.database.operations import get_db_session
from app.database.models import LinkedInJob, JobAnalysis
from app.config import X_AI_API_KEY, X_AI_BASE_URL

# 設置日誌
logger = logging.getLogger("linkedin_job_analyzer")

# 初始化 X.AI client
try:
    # 完整重構 OpenAI 客戶端初始化邏輯，避免 proxies 參數問題
    # 顯式指定所有必需的參數，避免使用 **params 可能引入不支援的參數
    # 新版 OpenAI SDK 不再支援 proxies 參數
    
    # 基本配置
    client = OpenAI(
        api_key=X_AI_API_KEY,
        base_url=X_AI_BASE_URL
    )
    
    logger.info("成功初始化 OpenAI 客戶端")
except Exception as e:
    logger.error(f"初始化 OpenAI 客戶端失敗: {str(e)}")
    # 如果出現錯誤，則嘗試不使用 base_url 重新初始化
    try:
        client = OpenAI(api_key=X_AI_API_KEY)
        logger.info("使用基本參數成功初始化 OpenAI 客戶端")
    except Exception as inner_e:
        logger.error(f"再次初始化 OpenAI 客戶端失敗: {str(inner_e)}")
        client = None
        
    # 即使初始化失敗，程式仍可繼續執行其他非 AI 相關功能

def get_unprocessed_jobs(hours_ago=1):
    """
    獲取最近爬取但尚未處理分析的職缺
    
    Args:
        hours_ago: 幾小時前的資料，預設為1小時
    
    Returns:
        pandas.DataFrame: 未處理的職缺資料
    """
    session = get_db_session()
    try:
        # 計算時間範圍
        cutoff_time = datetime.now() - timedelta(hours=hours_ago)
        
        # 查詢最近爬取但尚未分析過的職缺，只選取需要的欄位
        query = text("""
        SELECT j.id, j.company_name, j.job_title, j.description, j.link, j.posting_date,
               j.seniority_level, j.employment_type, j.job_function, j.industries, j.location, j.job_scrape
        FROM linkedin_jobs j
        LEFT JOIN job_analysis a ON j.id = a.job_id
        WHERE a.id IS NULL
        AND j.scrape_date >= :cutoff_time
        """)
        
        # 執行查詢
        result = session.execute(query, {"cutoff_time": cutoff_time})
        
        # 將結果轉為 DataFrame
        jobs = []
        for row in result:
            job_dict = {column: getattr(row, column) for column in row._mapping.keys()}
            jobs.append(job_dict)
        
        # 建立 DataFrame
        if jobs:
            df = pd.DataFrame(jobs)
            logger.info(f"找到 {len(df)} 筆未處理的職缺資料")
            return df
        else:
            logger.info("沒有找到需要處理的新職缺資料")
            return None
        
    except Exception as e:
        logger.error(f"獲取未處理職缺時發生錯誤: {str(e)}")
        return None
    finally:
        session.close()

def summarize_job_description(description, job_title, row_index):
    """
    使用 AI 分析職缺描述，提取關鍵資訊
    
    Args:
        description: 職缺描述
        job_title: 職位名稱
        row_index: 資料行索引
        
    Returns:
        dict: 分析結果
    """
    prompt = f"""Analyze the job title and description to extract:
- Keywords: AI/ML/DS-related terms (e.g., 'data, streaming')
- Hard skills: AI/ML/DS technical skills (e.g., 'kafka, python')
- Soft skills: Relevant to AI/ML/DS roles (e.g., 'teamwork, organization')
- Cleaned job title: Core role without qualifiers like 'senior' or 'location' (e.g., 'data engineer')
- Suitable personality: Traits suited for the role (e.g., 'analytical, organized')
- Representative anime character: Well-known from Japan/Taiwan, matching skills/personality (e.g., 'Edward Elric' or '艾德華·艾爾利克')
- Representative animal: Symbolizing the role's skills/traits, in English (e.g., 'beaver')
- Job superpower: Named ability from animation/movies/anime/TV, in Chinese (e.g., '鍊金術' from 'Fullmetal Alchemist')
- Job theme song: Well-known Chinese or English pop song (e.g., 'Shape of You' or '傻瓜')

Return as: 
{{
    'keywords': 'keyword1, keyword2',
    'hard_skill': 'skill1, skill2',
    'soft_skill': 'skill1, skill2',
    'cleaned_job_title': 'core title',
    'suitable_personality': 'trait1, trait2',
    'representative_anime_character': 'character name',
    'representative_animal': 'animal name',
    'job_superpower': 'superpower name',
    'job_theme_song': 'song name'
}}
- Use null if no info is found.
- Keywords, skills, and personality: Single-word, English, comma-separated.
- Cleaned job title: English, core role (e.g., 'Data Engineer' from 'Senior Data Engineer – Kafka Expert').
- Anime character: English or Chinese name, based on context (e.g., 'Edward Elric' or '艾德華·艾爾利克').
- Animal: English.
- Superpower: Chinese, from animation/movies/anime/TV (e.g., '鍊金術', '巨人變身').
- Theme song: Chinese or English pop song.

Examples:
- 'Senior Data Engineer – Kafka Expert' + 'manage Kafka streams, detail-oriented' → 
  'cleaned_job_title': 'data engineer', 'suitable_personality': 'analytical, organized', 
  'representative_anime_character': 'edward elric', 'representative_animal': 'beaver', 
  'job_superpower': '鍊金術', 'job_theme_song': 'shape of you'
- 'Machine Learning Engineer, Search E-Commerce' + 'innovative ML solutions' → 
  'cleaned_job_title': 'machine learning engineer', 'suitable_personality': 'creative, curious', 
  'representative_anime_character': 'erwin smith', 'representative_animal': 'owl', 
  'job_superpower': '巨人變身', 'job_theme_song': 'lemon'

Job Title: {job_title}
Job Description: {description}
"""
    
    try:
        completion = client.chat.completions.create(
            model="grok-2-latest",  # 使用 X.AI 的 grok-2 模型
            messages=[
                {"role": "system", "content": "You are an AI expert specializing in AI, Data Science, and Machine Learning, with deep knowledge of Japanese/Taiwanese anime, animation, movies, TV series, and Chinese/English pop music. Analyze job titles and descriptions to extract key information. Identify AI/ML/DS-related keywords, hard skills, and soft skills; clean the job title; infer personality traits; and select a representative anime character, animal, superpower (from animation, movies, anime, or TV series), and theme song (from Chinese or English pop music) that match the role's skills or personality. Return as a dictionary: {'keywords': 'keyword1, keyword2', 'hard_skill': 'skill1, skill2', 'soft_skill': 'skill1, skill2', 'cleaned_job_title': 'core title', 'suitable_personality': 'trait1, trait2', 'representative_anime_character': 'character name', 'representative_animal': 'animal name', 'job_superpower': 'superpower name', 'job_theme_song': 'song name'}. Use null if no info is found. Provide single-word lists for keywords, skills, and personality, prioritize AI/ML/DS content, and ensure lowercase terms with no duplicates."},
                {"role": "user", "content": prompt}
            ]
        )
        
        response = completion.choices[0].message.content
        
        try:
            result = eval(response)
            # Post-process list fields
            for key in ['keywords', 'hard_skill', 'soft_skill', 'suitable_personality']:
                if result[key] and isinstance(result[key], str):
                    words = set(word.lower() for phrase in result[key].split(', ') for word in phrase.split())
                    result[key] = ', '.join(sorted(words))
            # Post-process single-value fields
            for key in ['cleaned_job_title', 'representative_anime_character', 'representative_animal', 'job_superpower', 'job_theme_song']:
                if result.get(key) and isinstance(result[key], str):
                    result[key] = result[key].lower()
            result['row_index'] = row_index
            return result
        except:
            logger.error(f"無法解析 AI 回應: {response}")
            return get_default_result(row_index)
            
    except Exception as e:
        logger.error(f"AI 分析過程中發生錯誤: {str(e)}")
        return get_default_result(row_index)

def get_default_result(row_index):
    """返回預設的空結果"""
    return {
        'keywords': None,
        'hard_skill': None,
        'soft_skill': None,
        'cleaned_job_title': None,
        'suitable_personality': None,
        'representative_anime_character': None,
        'representative_animal': None,
        'job_superpower': None,
        'job_theme_song': None,
        'row_index': row_index
    }

def process_job_descriptions(df):
    """
    處理一批職缺描述
    
    Args:
        df: 職缺資料 DataFrame
        
    Returns:
        pandas.DataFrame: 處理後的結果
    """
    if df is None or len(df) == 0:
        logger.info("沒有資料需要處理")
        return None
        
    results = []
    total_rows = len(df)
    
    # 處理每一筆職缺資料並保存索引以便後續合併
    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Processing job descriptions"):
        logger.info(f"處理 {idx + 1}/{total_rows}: {row['company_name']} - {row['job_title']}")
        summary = summarize_job_description(row['description'], row['job_title'], idx)
        results.append(summary)
    
    # 將分析結果轉為 DataFrame
    summary_df = pd.DataFrame(results)
    
    # 合併原始職缺資料與分析結果
    result_df = pd.merge(
        df,
        summary_df[['row_index', 'keywords', 'hard_skill', 'soft_skill', 'cleaned_job_title', 
                    'suitable_personality', 'representative_anime_character', 'representative_animal', 
                    'job_superpower', 'job_theme_song']],
        left_index=True,
        right_on='row_index',
        how='left'
    ).drop('row_index', axis=1)
    
    logger.info(f"完成處理 {total_rows} 筆職缺描述")
    return result_df

def save_analysis_results(result_df):
    """
    將分析結果儲存到資料庫
    
    Args:
        result_df: 處理後的 DataFrame
        
    Returns:
        int: 成功儲存的記錄數
    """
    if result_df is None or len(result_df) == 0:
        logger.info("沒有結果需要儲存")
        return 0
        
    session = get_db_session()
    saved_count = 0
    
    try:
        for _, row in result_df.iterrows():
            # 使用直接從查詢中獲取的相同欄位值，確保完整性
            analysis = JobAnalysis(
                job_id=row['id'],
                keywords=row['keywords'],
                hard_skill=row['hard_skill'],
                soft_skill=row['soft_skill'],
                cleaned_job_title=row['cleaned_job_title'],
                suitable_personality=row['suitable_personality'],
                representative_anime_character=row['representative_anime_character'],
                representative_animal=row['representative_animal'],
                job_superpower=row['job_superpower'],
                job_theme_song=row['job_theme_song'],
                processed_at=datetime.now()
            )
            session.add(analysis)
            saved_count += 1
            
        session.commit()
        logger.info(f"成功儲存 {saved_count} 筆分析結果")
        return saved_count
        
    except Exception as e:
        session.rollback()
        logger.error(f"儲存分析結果時發生錯誤: {str(e)}")
        return 0
    finally:
        session.close()

def run_job_analysis(hours_ago=1, batch_size=100):
    """
    運行職缺分析主流程
    
    Args:
        hours_ago: 分析幾小時前的資料，預設為1小時
        batch_size: 一次處理的批次大小，避免記憶體過載
        
    Returns:
        bool: 是否成功完成分析
    """
    logger.info(f"開始執行職缺分析，處理 {hours_ago} 小時前的資料")
    
    try:
        # 1. 獲取未處理的職缺資料
        jobs_df = get_unprocessed_jobs(hours_ago)
        
        if jobs_df is None or len(jobs_df) == 0:
            logger.info("沒有找到需要處理的新職缺資料")
            return True
        
        total_jobs = len(jobs_df)
        total_processed = 0
        
        # 依據批次大小處理資料
        for start_idx in range(0, total_jobs, batch_size):
            end_idx = min(start_idx + batch_size, total_jobs)
            batch_df = jobs_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
            
            logger.info(f"處理批次 {start_idx//batch_size + 1}/{(total_jobs+batch_size-1)//batch_size}，資料範圍: {start_idx+1}-{end_idx} / {total_jobs}")
            
            # 2. 處理這一批職缺描述
            processed_df = process_job_descriptions(batch_df)
            
            # 3. 儲存這一批分析結果
            saved_count = save_analysis_results(processed_df)
            total_processed += saved_count
            
            logger.info(f"批次處理完成，儲存 {saved_count} 筆分析結果")
        
        logger.info(f"職缺分析全部完成，共處理 {total_jobs} 筆資料，成功儲存 {total_processed} 筆結果")
        return True
        
    except Exception as e:
        logger.error(f"執行職缺分析時發生錯誤: {str(e)}")
        return False

# 測試用的主函數
if __name__ == "__main__":
    # 設置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 運行分析
    run_job_analysis(hours_ago=24, batch_size=50)  # 處理最近24小時的資料，以50筆為一批次 