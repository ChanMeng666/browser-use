import asyncio
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from browser_use import Agent, Browser, BrowserConfig

# Load environment variables
load_dotenv()

# Initialize the model
api_key = os.getenv('GEMINI_API_KEY')
linkedin_email = os.getenv('LINKEDIN_EMAIL')
linkedin_password = os.getenv('LINKEDIN_PASSWORD')

# 配置浏览器设置
browser_config = BrowserConfig(
    headless=False,  # 设置为False以查看浏览器操作
    disable_security=True
)

# 初始化浏览器
browser = Browser(config=browser_config)

# Initialize the model with specific parameters
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-exp',
    api_key=SecretStr(api_key),
    temperature=0.0,  # 降低随机性
    max_output_tokens=2048  # 增加输出长度限制
)

async def main():
    # Create agent with the model
    login_agent = Agent(
        task=f"Go to LinkedIn login page and login with email: {linkedin_email} and password: {linkedin_password}. Navigate to LinkedIn's My Network page, locate the Catch up section, and send celebratory messages by repeatedly doing the following: Find each button containing either 'Congrats' or 'Happy' in the Catch up section, click it to open the popup modal, then click the 'Send' button in the modal. If 'Message sent' appears for all visible updates, scroll down the page to find more 'Congrats' or 'Happy' buttons. Continue this process until no more actionable buttons are available in the Catch up section, even after scrolling.",
        llm=llm,
        browser=browser,
        use_vision=True
    )
    await login_agent.run()
    
    # Create agent for the main task
    agent = Agent(
        task="Navigate to LinkedIn's My Network page, locate the Catch up section, and send congratulatory messages by repeatedly doing the following: Find each 'Congrats' button in the Catch up section, click it to open the popup modal, then click the 'Send' button in the modal. Continue this process until no more 'Congrats' buttons are available in the Catch up section.",
        llm=llm,
        browser=browser,  # 使用配置好的浏览器实例
        use_vision=True  # 启用视觉能力
    )
    result = await agent.run()
    print(result)
    
    # 完成后关闭浏览器
    await browser.close()

asyncio.run(main())
