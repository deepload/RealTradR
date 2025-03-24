"""
Sentiment Analyzer for RealTradR

This module provides functionality to analyze sentiment from various sources:
- Twitter/X data
- Reddit posts
- Financial news articles
- Company reports

The sentiment scores are used as additional signals for the trading strategy.
"""

import os
import re
import json
import logging
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sqlalchemy.orm import Session

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API Keys
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Download NLTK data if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')


class SentimentAnalyzer:
    """Class for analyzing sentiment from various sources"""
    
    def __init__(self, db=None):
        """
        Initialize the sentiment analyzer
        
        Args:
            db: Optional database session
        """
        self.db = db
        self.vader = SentimentIntensityAnalyzer()
        logger.info("Initialized SentimentAnalyzer")
    
    def analyze_text(self, text):
        """
        Analyze sentiment of a text string
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text or not isinstance(text, str):
            return {
                "compound": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0
            }
        
        # Clean text
        text = self._clean_text(text)
        
        # Get VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        
        # Get TextBlob sentiment
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Combine scores (weighted average)
        compound = vader_scores['compound'] * 0.7 + textblob_polarity * 0.3
        
        return {
            "compound": compound,
            "positive": vader_scores['pos'],
            "negative": vader_scores['neg'],
            "neutral": vader_scores['neu'],
            "subjectivity": textblob_subjectivity
        }
    
    def _clean_text(self, text):
        """
        Clean text for sentiment analysis
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags symbol (but keep the text)
        text = re.sub(r'#', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_twitter_sentiment(self, symbol, days=3, count=100):
        """
        Get sentiment from Twitter/X for a symbol
        
        Args:
            symbol: Stock symbol to analyze
            days: Number of days to look back
            count: Maximum number of tweets to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not TWITTER_BEARER_TOKEN:
            logger.warning("Twitter API token not configured")
            return None
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Prepare search query
            company_name = self._get_company_name(symbol)
            query = f"({symbol} OR ${symbol}"
            if company_name:
                query += f" OR {company_name}"
            query += ") lang:en -is:retweet"
            
            # Twitter API v2 endpoint
            url = "https://api.twitter.com/2/tweets/search/recent"
            
            headers = {
                "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"
            }
            
            params = {
                "query": query,
                "max_results": min(count, 100),  # API limit is 100 per request
                "tweet.fields": "created_at,public_metrics",
                "start_time": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_time": end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code != 200:
                logger.error(f"Twitter API error: {response.status_code} - {response.text}")
                return None
            
            data = response.json()
            
            if "data" not in data or not data["data"]:
                logger.info(f"No tweets found for {symbol}")
                return {
                    "compound": 0,
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                    "tweet_count": 0
                }
            
            tweets = data["data"]
            logger.info(f"Found {len(tweets)} tweets for {symbol}")
            
            # Analyze sentiment for each tweet
            sentiments = []
            for tweet in tweets:
                sentiment = self.analyze_text(tweet["text"])
                # Weight by engagement (likes + retweets)
                engagement = 1
                if "public_metrics" in tweet:
                    likes = tweet["public_metrics"].get("like_count", 0)
                    retweets = tweet["public_metrics"].get("retweet_count", 0)
                    engagement = max(1, likes + retweets)
                
                # Apply time decay - more recent tweets have higher weight
                created_at = datetime.strptime(tweet["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
                time_diff = (end_date - created_at).total_seconds() / 86400  # days
                time_weight = max(0.5, 1 - (time_diff / (days * 1.5)))
                
                # Final weight
                weight = engagement * time_weight
                sentiments.append((sentiment, weight))
            
            # Calculate weighted average
            if not sentiments:
                return {
                    "compound": 0,
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                    "tweet_count": 0
                }
            
            total_weight = sum(weight for _, weight in sentiments)
            
            weighted_compound = sum(s["compound"] * w for s, w in sentiments) / total_weight
            weighted_positive = sum(s["positive"] * w for s, w in sentiments) / total_weight
            weighted_negative = sum(s["negative"] * w for s, w in sentiments) / total_weight
            weighted_neutral = sum(s["neutral"] * w for s, w in sentiments) / total_weight
            
            return {
                "compound": weighted_compound,
                "positive": weighted_positive,
                "negative": weighted_negative,
                "neutral": weighted_neutral,
                "tweet_count": len(tweets)
            }
            
        except Exception as e:
            logger.error(f"Error getting Twitter sentiment for {symbol}: {e}")
            return None
    
    def get_reddit_sentiment(self, symbol, subreddits=None, days=3, limit=100):
        """
        Get sentiment from Reddit for a symbol
        
        Args:
            symbol: Stock symbol to analyze
            subreddits: List of subreddits to search (default: investing, stocks, wallstreetbets)
            days: Number of days to look back
            limit: Maximum number of posts to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
            logger.warning("Reddit API credentials not configured")
            return None
        
        if subreddits is None:
            subreddits = ["investing", "stocks", "wallstreetbets"]
        
        try:
            # Get Reddit authentication token
            auth = requests.auth.HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
            data = {
                'grant_type': 'client_credentials',
                'username': os.getenv("REDDIT_USERNAME", ""),
                'password': os.getenv("REDDIT_PASSWORD", "")
            }
            headers = {'User-Agent': 'RealTradR/0.1'}
            
            response = requests.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=auth,
                data=data,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"Reddit authentication error: {response.status_code} - {response.text}")
                return None
            
            token = response.json().get('access_token')
            headers = {**headers, **{'Authorization': f"bearer {token}"}}
            
            # Calculate timestamp for filtering by date
            timestamp = int((datetime.now() - timedelta(days=days)).timestamp())
            
            all_posts = []
            all_comments = []
            
            # Get posts from each subreddit
            for subreddit in subreddits:
                # Search for posts containing the symbol
                search_url = f"https://oauth.reddit.com/r/{subreddit}/search"
                params = {
                    'q': f"{symbol} OR ${symbol}",
                    'sort': 'relevance',
                    'restrict_sr': 'on',
                    'limit': limit,
                    't': 'week'  # time: hour, day, week, month, year, all
                }
                
                response = requests.get(search_url, headers=headers, params=params)
                
                if response.status_code != 200:
                    logger.error(f"Reddit API error for r/{subreddit}: {response.status_code}")
                    continue
                
                data = response.json()
                posts = [p['data'] for p in data['data']['children'] if p['data']['created_utc'] > timestamp]
                
                # Filter posts that actually mention the symbol (to avoid false positives)
                company_name = self._get_company_name(symbol)
                filtered_posts = []
                for post in posts:
                    title = post.get('title', '').lower()
                    selftext = post.get('selftext', '').lower()
                    
                    if (symbol.lower() in title or f"${symbol.lower()}" in title or 
                        symbol.lower() in selftext or f"${symbol.lower()}" in selftext or
                        (company_name and company_name.lower() in title) or
                        (company_name and company_name.lower() in selftext)):
                        filtered_posts.append(post)
                
                all_posts.extend(filtered_posts)
                
                # Get comments for top posts
                for post in filtered_posts[:10]:  # Limit to top 10 posts to avoid too many requests
                    post_id = post['id']
                    comment_url = f"https://oauth.reddit.com/r/{subreddit}/comments/{post_id}"
                    
                    response = requests.get(comment_url, headers=headers)
                    
                    if response.status_code != 200:
                        continue
                    
                    comments_data = response.json()
                    if len(comments_data) > 1:
                        comments = [c['data'] for c in comments_data[1]['data']['children'] 
                                   if c['kind'] == 't1' and c['data']['created_utc'] > timestamp]
                        all_comments.extend(comments)
            
            logger.info(f"Found {len(all_posts)} posts and {len(all_comments)} comments for {symbol}")
            
            # Analyze sentiment for posts
            post_sentiments = []
            for post in all_posts:
                # Combine title and body text
                text = post.get('title', '') + ' ' + post.get('selftext', '')
                sentiment = self.analyze_text(text)
                
                # Weight by score and number of comments
                score = post.get('score', 0)
                num_comments = post.get('num_comments', 0)
                weight = max(1, score + num_comments)
                
                post_sentiments.append((sentiment, weight))
            
            # Analyze sentiment for comments
            comment_sentiments = []
            for comment in all_comments:
                text = comment.get('body', '')
                sentiment = self.analyze_text(text)
                
                # Weight by score
                score = comment.get('score', 0)
                weight = max(1, score)
                
                comment_sentiments.append((sentiment, weight))
            
            # Combine post and comment sentiments
            all_sentiments = post_sentiments + comment_sentiments
            
            if not all_sentiments:
                return {
                    "compound": 0,
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                    "post_count": 0,
                    "comment_count": 0
                }
            
            total_weight = sum(weight for _, weight in all_sentiments)
            
            weighted_compound = sum(s["compound"] * w for s, w in all_sentiments) / total_weight
            weighted_positive = sum(s["positive"] * w for s, w in all_sentiments) / total_weight
            weighted_negative = sum(s["negative"] * w for s, w in all_sentiments) / total_weight
            weighted_neutral = sum(s["neutral"] * w for s, w in all_sentiments) / total_weight
            
            return {
                "compound": weighted_compound,
                "positive": weighted_positive,
                "negative": weighted_negative,
                "neutral": weighted_neutral,
                "post_count": len(all_posts),
                "comment_count": len(all_comments)
            }
            
        except Exception as e:
            logger.error(f"Error getting Reddit sentiment for {symbol}: {e}")
            return None
    
    def get_news_sentiment(self, symbol, days=3):
        """
        Get sentiment from financial news for a symbol
        
        Args:
            symbol: Stock symbol to analyze
            days: Number of days to look back
            
        Returns:
            Dictionary with sentiment scores
        """
        if not NEWS_API_KEY:
            logger.warning("News API key not configured")
            return None
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get company name for better search results
            company_name = self._get_company_name(symbol)
            query = symbol
            if company_name:
                query = f"{company_name} OR {symbol}"
            
            # News API endpoint
            url = "https://newsapi.org/v2/everything"
            
            params = {
                "q": query,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "language": "en",
                "sortBy": "relevancy",
                "apiKey": NEWS_API_KEY
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                logger.error(f"News API error: {response.status_code} - {response.text}")
                return None
            
            data = response.json()
            
            if "articles" not in data or not data["articles"]:
                logger.info(f"No news articles found for {symbol}")
                return {
                    "compound": 0,
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                    "article_count": 0
                }
            
            articles = data["articles"]
            logger.info(f"Found {len(articles)} news articles for {symbol}")
            
            # Analyze sentiment for each article
            sentiments = []
            for article in articles:
                # Combine title and description
                text = article.get("title", "") + " " + article.get("description", "")
                sentiment = self.analyze_text(text)
                
                # Apply time decay - more recent articles have higher weight
                published_at = datetime.strptime(article["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
                time_diff = (end_date - published_at).total_seconds() / 86400  # days
                weight = max(0.5, 1 - (time_diff / (days * 1.5)))
                
                sentiments.append((sentiment, weight))
            
            if not sentiments:
                return {
                    "compound": 0,
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                    "article_count": 0
                }
            
            total_weight = sum(weight for _, weight in sentiments)
            
            weighted_compound = sum(s["compound"] * w for s, w in sentiments) / total_weight
            weighted_positive = sum(s["positive"] * w for s, w in sentiments) / total_weight
            weighted_negative = sum(s["negative"] * w for s, w in sentiments) / total_weight
            weighted_neutral = sum(s["neutral"] * w for s, w in sentiments) / total_weight
            
            return {
                "compound": weighted_compound,
                "positive": weighted_positive,
                "negative": weighted_negative,
                "neutral": weighted_neutral,
                "article_count": len(articles)
            }
            
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {e}")
            return None
    
    def get_combined_sentiment(self, symbol, days=3):
        """
        Get combined sentiment from all sources
        
        Args:
            symbol: Stock symbol to analyze
            days: Number of days to look back
            
        Returns:
            Dictionary with combined sentiment scores
        """
        # Get sentiment from each source
        twitter_sentiment = self.get_twitter_sentiment(symbol, days)
        reddit_sentiment = self.get_reddit_sentiment(symbol, days=days)
        news_sentiment = self.get_news_sentiment(symbol, days)
        
        # Default values if any source fails
        twitter_compound = twitter_sentiment["compound"] if twitter_sentiment else 0
        reddit_compound = reddit_sentiment["compound"] if reddit_sentiment else 0
        news_compound = news_sentiment["compound"] if news_sentiment else 0
        
        # Calculate weighted average
        # News is most reliable, then Reddit, then Twitter
        weights = {
            "twitter": 0.2,
            "reddit": 0.3,
            "news": 0.5
        }
        
        # Adjust weights if any source is missing
        total_weight = sum(weights.values())
        if twitter_sentiment is None:
            weights["reddit"] += weights["twitter"] * weights["reddit"] / (weights["reddit"] + weights["news"])
            weights["news"] += weights["twitter"] * weights["news"] / (weights["reddit"] + weights["news"])
            weights["twitter"] = 0
        
        if reddit_sentiment is None:
            weights["twitter"] += weights["reddit"] * weights["twitter"] / (weights["twitter"] + weights["news"])
            weights["news"] += weights["reddit"] * weights["news"] / (weights["twitter"] + weights["news"])
            weights["reddit"] = 0
        
        if news_sentiment is None:
            weights["twitter"] += weights["news"] * weights["twitter"] / (weights["twitter"] + weights["reddit"])
            weights["reddit"] += weights["news"] * weights["reddit"] / (weights["twitter"] + weights["reddit"])
            weights["news"] = 0
        
        # Normalize weights
        new_total = sum(weights.values())
        if new_total > 0:
            for k in weights:
                weights[k] = weights[k] / new_total
        
        # Calculate combined sentiment
        combined_compound = (
            twitter_compound * weights["twitter"] +
            reddit_compound * weights["reddit"] +
            news_compound * weights["news"]
        )
        
        # Create result dictionary
        result = {
            "compound": combined_compound,
            "twitter_compound": twitter_compound,
            "reddit_compound": reddit_compound,
            "news_compound": news_compound,
            "twitter_weight": weights["twitter"],
            "reddit_weight": weights["reddit"],
            "news_weight": weights["news"]
        }
        
        # Add source-specific details
        if twitter_sentiment:
            result["twitter_details"] = {
                "positive": twitter_sentiment["positive"],
                "negative": twitter_sentiment["negative"],
                "neutral": twitter_sentiment["neutral"],
                "tweet_count": twitter_sentiment.get("tweet_count", 0)
            }
        
        if reddit_sentiment:
            result["reddit_details"] = {
                "positive": reddit_sentiment["positive"],
                "negative": reddit_sentiment["negative"],
                "neutral": reddit_sentiment["neutral"],
                "post_count": reddit_sentiment.get("post_count", 0),
                "comment_count": reddit_sentiment.get("comment_count", 0)
            }
        
        if news_sentiment:
            result["news_details"] = {
                "positive": news_sentiment["positive"],
                "negative": news_sentiment["negative"],
                "neutral": news_sentiment["neutral"],
                "article_count": news_sentiment.get("article_count", 0)
            }
        
        # Save to database if available
        self._save_sentiment_to_db(symbol, result)
        
        return result
    
    def _get_company_name(self, symbol):
        """
        Get company name for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Company name or None
        """
        # Try to get from database first
        if self.db:
            from app.models.trading_symbol import TradingSymbol
            
            symbol_obj = self.db.query(TradingSymbol).filter(TradingSymbol.symbol == symbol).first()
            if symbol_obj and symbol_obj.name:
                return symbol_obj.name
        
        # Fallback to hardcoded mapping for common symbols
        symbol_map = {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Google",
            "GOOG": "Google",
            "AMZN": "Amazon",
            "META": "Meta",
            "TSLA": "Tesla",
            "NVDA": "NVIDIA",
            "JPM": "JPMorgan",
            "V": "Visa",
            "JNJ": "Johnson & Johnson",
            "WMT": "Walmart",
            "PG": "Procter & Gamble",
            "MA": "Mastercard",
            "UNH": "UnitedHealth",
            "HD": "Home Depot",
            "BAC": "Bank of America",
            "DIS": "Disney",
            "NFLX": "Netflix",
            "PYPL": "PayPal"
        }
        
        return symbol_map.get(symbol.upper())
    
    def _save_sentiment_to_db(self, symbol, sentiment_data):
        """
        Save sentiment data to database
        
        Args:
            symbol: Stock symbol
            sentiment_data: Sentiment data dictionary
        """
        if not self.db:
            return
        
        try:
            from app.models.trading_symbol import TradingSymbol
            from app.models.sentiment import NewsSentiment
            
            # Get symbol ID
            symbol_obj = self.db.query(TradingSymbol).filter(TradingSymbol.symbol == symbol).first()
            if not symbol_obj:
                logger.warning(f"Symbol {symbol} not found in database")
                return
            
            # Create sentiment record
            sentiment = NewsSentiment(
                symbol_id=symbol_obj.id,
                compound_score=sentiment_data["compound"],
                positive_score=sentiment_data.get("positive", 0),
                negative_score=sentiment_data.get("negative", 0),
                neutral_score=sentiment_data.get("neutral", 0),
                source_data=json.dumps(sentiment_data),
                created_at=datetime.utcnow()
            )
            
            self.db.add(sentiment)
            self.db.commit()
            
            logger.info(f"Saved sentiment data for {symbol}")
            
        except Exception as e:
            logger.error(f"Error saving sentiment data: {e}")
            if self.db:
                self.db.rollback()


# Function to get sentiment for a symbol
def get_symbol_sentiment(symbol, days=3, db=None):
    """
    Get sentiment for a symbol
    
    Args:
        symbol: Stock symbol
        days: Number of days to look back
        db: Optional database session
        
    Returns:
        Dictionary with sentiment scores
    """
    analyzer = SentimentAnalyzer(db)
    return analyzer.get_combined_sentiment(symbol, days)
