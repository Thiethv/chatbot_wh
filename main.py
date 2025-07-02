# Enhanced main.py with voice and advanced features
import flet as ft
import asyncio
import aiohttp
import speech_recognition as sr
import pyttsx3
from typing import Dict, List
import json
import threading

class AdvancedWarehouseChatbot:
    def __init__(self):
        self.api_base_url = "http://localhost:8000/api/v1"
        self.chat_history = []
        self.voice_enabled = True
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.is_listening = False

        self.page = None
        
    async def main(self, page: ft.Page):
        page.title = "Trợ lý kho ảo - Nâng cao"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.padding = 20
        page.window_width = 1200
        page.window_height = 800

        self.page = page
        
        # Initialize TTS
        self.tts_engine.setProperty('rate', 150)
        voices = self.tts_engine.getProperty('voices')
        if voices:
            self.tts_engine.setProperty('voice', voices[0].id)
        
        # Create layout components
        await self.create_layout(page)
        
        # Load initial suggestions
        await self.load_suggestions()
        
        # Welcome message
        await self.add_message(
            "🤖 Chào bạn! Tôi là trợ lý kho ảo thông minh.\n"
            "✨ Tính năng mới: Hỗ trợ giọng nói, tìm kiếm thông minh, gợi ý tự động\n"
            "📝 Bạn có thể hỏi về mã hàng, quy trình, vị trí kho...", 
            "bot"
        )
    
    async def create_layout(self, page: ft.Page):
        """Create advanced UI layout"""
        
        # Header
        header = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.WAREHOUSE, size=40, color=ft.Colors.BLUE),
                ft.Text("Warehouse AI Assistant", size=36, weight=ft.FontWeight.BOLD),
                ft.Container(expand=True)
            ]),
            padding=20,
            bgcolor=ft.Colors.BLUE_50,
            border_radius=10
        )
        
        # Main content area
        content_area = ft.Row([
            # Left sidebar - Quick actions
            ft.Container(
                content=await self.create_sidebar(),
                width=300,
                bgcolor=ft.Colors.GREY_50,
                padding=20,
                border_radius=10
            ),
            
            # Center - Chat area
            ft.Container(
                content=await self.create_chat_area(),
                expand=True,
                bgcolor=ft.Colors.WHITE,
                padding=20,
                border_radius=10,
                margin=ft.margin.only(left=10, right=10)
            ),
            
            # Right sidebar - Context info
            ft.Container(
                content=await self.create_context_panel(),
                width=300,
                bgcolor=ft.Colors.GREY_50,
                padding=20,
                border_radius=10
            )
        ])
        
        # Add to page
        page.add(
            ft.Column([
                header,
                ft.Container(height=10),
                content_area
            ])
        )
    
    async def create_sidebar(self):
        """Create left sidebar with quick actions"""
        
        # Category buttons
        categories = [
            {"name": "Tìm hàng", "icon": "🔍", "query": "tìm kiếm sản phẩm"},
            {"name": "Quy trình", "icon": "📋", "query": "quy trình làm việc"},
            {"name": "Vị trí kho", "icon": "📍", "query": "vị trí lưu kho"},
            {"name": "Tồn kho", "icon": "📊", "query": "báo cáo tồn kho"},
            {"name": "Nhập/Xuất", "icon": "📦", "query": "lịch sử giao dịch"},
            {"name": "Hướng dẫn", "icon": "❓", "query": "hướng dẫn sử dụng"}
        ]
        
        category_buttons = []
        for cat in categories:
            btn = ft.ElevatedButton(
                content=ft.Row([
                    ft.Text(cat["icon"], size=20),
                    ft.Text(cat["name"], expand=True)
                ]),
                width=250,
                height=50,
                on_click=lambda e, query=cat["query"]: asyncio.create_task(
                    self.send_message_programmatically(query)
                )
            )
            category_buttons.append(btn)
        
        # Recent queries
        self.recent_queries_list = ft.Column([
            ft.Text("📝 Câu hỏi gần đây", size=16, weight=ft.FontWeight.W_500)
        ])
        
        return ft.Column([
            ft.Text("🚀 Thao tác nhanh", size=18, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            *category_buttons,
            ft.Divider(),
            self.recent_queries_list
        ])
    
    async def create_chat_area(self):
        """Create main chat area"""
        
        # Chat messages container
        self.chat_container = ft.Column(
            controls=[],
            scroll=ft.ScrollMode.AUTO,
            auto_scroll=True,
            height=400,
            expand=True
        )
        
        # Input area
        self.input_field = ft.TextField(
            hint_text="Nhập câu hỏi hoặc nhấn mic để nói...",
            expand=True,
            multiline=True,
            max_lines=3,
            filled=True,
            shift_enter=True,
            on_submit=self.send_message,
            border_radius=10
        )
        
        # Action buttons
        input_actions = ft.Row([
            ft.IconButton(
                icon=ft.Icons.MIC,
                tooltip="Nói",
                on_click=self.start_voice_input,
                bgcolor=ft.Colors.GREEN_100
            ),
            ft.IconButton(
                icon=ft.Icons.PHOTO_CAMERA,
                tooltip="Chụp ảnh mã vạch",
                on_click=self.scan_barcode
            )
        ])
        
        return ft.Column([
            ft.Text("💬 Trò chuyện", size=18, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ft.Container(
                content=self.chat_container,
                border=ft.border.all(1, ft.Colors.GREY_300),
                border_radius=10,
                padding=10,
                bgcolor=ft.Colors.GREY_50
            ),
            ft.Container(height=10),
            self.input_field,
            input_actions
        ])
    
    async def create_context_panel(self):
        """Create right context panel"""
        
        self.context_panel = ft.Column([
            ft.Text("📊 Thông tin ngữ cảnh", size=18, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ft.Text("Sẽ hiển thị thông tin liên quan khi bạn đặt câu hỏi", 
                   color=ft.Colors.GREY_600)
        ])
        
        return self.context_panel
    
    async def send_message(self, e=None):
        """Enhanced send message with context updates"""
        if not self.input_field.value.strip():
            return
            
        user_message = self.input_field.value.strip()
        self.input_field.value = ""
        self.input_field.update()
        
        # Add to recent queries
        await self.add_recent_query(user_message)
        
        # Add user message
        await self.add_message(user_message, "user")
        
        # Show typing indicator
        typing_msg = await self.add_message("🤔 Đang suy nghĩ...", "bot")
        
        try:
            # Call API
            response = await self.call_chat_api(user_message)
            
            # Remove typing indicator
            self.chat_container.controls.remove(typing_msg)
            
            # Add bot response
            await self.add_message(response["answer"], "bot")
            
            # Update context panel
            await self.update_context_panel(response.get("context", {}))
            
            # Add sources if available
            if response.get("sources"):
                sources_text = "📚 Nguồn tham khảo:\n" + "\n".join([
                    f"• {source.get('metadata', {}).get('title', 'Tài liệu')}"
                    for source in response["sources"][:3]
                ])
                await self.add_message(sources_text, "info")
            
            # Add suggestions if available
            if response.get("suggestions"):
                suggestions_text = "💡 Gợi ý câu hỏi tiếp theo:\n" + "\n".join([
                    f"• {suggestion}" for suggestion in response["suggestions"][:3]
                ])
                await self.add_message(suggestions_text, "suggestion")
            
            # Text-to-speech if enabled
            if self.voice_enabled:
                await self.speak_text(response["answer"])
                
        except Exception as error:
            # Remove typing indicator
            if typing_msg in self.chat_container.controls:
                self.chat_container.controls.remove(typing_msg)
            await self.add_message(f"❌ Lỗi: {str(error)}", "error")
    
    async def add_message(self, message: str, sender: str):
        """Enhanced message display with better formatting"""
        Colors = {
            "user": ft.Colors.BLUE_50,
            "bot": ft.Colors.GREEN_50,
            "info": ft.Colors.ORANGE_50,
            "error": ft.Colors.RED_50,
            "suggestion": ft.Colors.PURPLE_50
        }
        
        Icons = {
            "user": "👤",
            "bot": "🤖",
            "info": "ℹ️",
            "error": "❌",
            "suggestion": "💡"
        }
        
        # Create interactive suggestions
        if sender == "suggestion":
            suggestion_buttons = []
            suggestions = message.split("\n")[1:]  # Skip header
            for suggestion in suggestions:
                if suggestion.strip().startswith("•"):
                    clean_suggestion = suggestion.strip()[1:].strip()
                    btn = ft.TextButton(
                        clean_suggestion,
                        on_click=lambda e, text=clean_suggestion: asyncio.create_task(
                            self.send_message_programmatically(text)
                        )
                    )
                    suggestion_buttons.append(btn)
            
            message_content = ft.Column([
                ft.Text("💡 Gợi ý câu hỏi tiếp theo:", weight=ft.FontWeight.W_500),
                ft.Column(suggestion_buttons)
            ])
        else:
            message_content = ft.Row([
                ft.Text(Icons.get(sender, ""), size=20),
                ft.Text(message, expand=True, selectable=True)
            ])
        
        message_container = ft.Container(
            content=message_content,
            bgcolor=Colors.get(sender, ft.Colors.GREY_50),
            padding=15,
            border_radius=10,
            margin=ft.margin.only(bottom=10),
            animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT)
        )
        
        self.chat_container.controls.append(message_container)
        self.chat_container.update()
        
        return message_container
    
    async def start_voice_input(self, e=None):
        """Start voice recognition"""
        if self.is_listening:
            return
        
        self.is_listening = True
        
        # Update UI
        await self.add_message("🎤 Đang nghe... Nói câu hỏi của bạn", "info")
        
        loop = asyncio.get_running_loop()  # Lấy event loop hiện tại
        def voice_thread():
            try:
                with sr.Microphone() as source:
                    # Adjust for ambient noise
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    
                    # Listen for audio
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    
                    # Recognize speech using Google Speech Recognition
                    text = self.recognizer.recognize_google(audio, language='vi-VN')
                    
                    # Update input field
                    asyncio.run_coroutine_threadsafe(self.handle_voice_result(text), loop)
                    
            except sr.UnknownValueError:
                asyncio.run_coroutine_threadsafe(self.handle_voice_error("Không thể nhận diện giọng nói"), loop)
            except sr.RequestError as e:
                asyncio.run_coroutine_threadsafe(self.handle_voice_error(f"Lỗi dịch vụ: {e}"), loop)
            except sr.WaitTimeoutError:
                asyncio.run_coroutine_threadsafe(self.handle_voice_error("Hết thời gian chờ"), loop)
            finally:
                self.is_listening = False
        
        # Run in separate thread
        threading.Thread(target=voice_thread, daemon=True).start()
    
    async def handle_voice_result(self, text: str):
        """Handle successful voice recognition"""
        await self.add_message(f"🎤 Đã nghe: {text}", "info")
        self.input_field.value = text
        self.input_field.update()
        await self.send_message()
    
    async def handle_voice_error(self, error_msg: str):
        """Handle voice recognition error"""
        await self.add_message(f"🎤 {error_msg}", "error")
    
    async def speak_text(self, text: str):
        """Text-to-speech"""
        def tts_thread():
            try:
                # Clean text for TTS
                clean_text = text.replace("•", "").replace("📚", "").replace("💡", "")
                self.tts_engine.say(clean_text[:200])  # Limit length
                self.tts_engine.runAndWait()
            except:
                pass
        
        threading.Thread(target=tts_thread, daemon=True).start()
    
    async def scan_barcode(self, e=None):
        """Simulate barcode scanning (would integrate with camera)"""
        await self.add_message("📷 Tính năng quét mã vạch đang được phát triển...", "info")
       
    async def update_context_panel(self, context: dict):
        """Update right context panel with relevant information"""
        if not context:
            return
        
        context_items = []
        
        # Products context
        if context.get("products"):
            context_items.append(ft.Text("📦 Sản phẩm liên quan:", weight=ft.FontWeight.W_500))
            for product in context["products"][:3]:
                context_items.append(
                    ft.Container(
                        content=ft.Column([
                            ft.Text(f"• {product['name']}", size=12),
                            ft.Text(f"  Mã: {product['code']}", size=10, color=ft.Colors.GREY_600),
                            ft.Text(f"  Tồn: {product.get('current_stock', 0)}", size=10, color=ft.Colors.GREY_600)
                        ]),
                        padding=5,
                        bgcolor=ft.Colors.BLUE_50,
                        border_radius=5,
                        margin=ft.margin.only(bottom=5)
                    )
                )
        
        # Locations context
        if context.get("locations"):
            context_items.append(ft.Text("📍 Vị trí liên quan:", weight=ft.FontWeight.W_500))
            for location in context["locations"][:3]:
                context_items.append(
                    ft.Container(
                        content=ft.Text(f"• {location['location_code']}", size=12),
                        padding=5,
                        bgcolor=ft.Colors.GREEN_50,
                        border_radius=5,
                        margin=ft.margin.only(bottom=5)
                    )
                )
        
        # Procedures context
        if context.get("procedures"):
            context_items.append(ft.Text("📋 Quy trình liên quan:", weight=ft.FontWeight.W_500))
            for procedure in context["procedures"][:2]:
                context_items.append(
                    ft.Container(
                        content=ft.Text(f"• {procedure['title']}", size=12),
                        padding=5,
                        bgcolor=ft.Colors.ORANGE_50,
                        border_radius=5,
                        margin=ft.margin.only(bottom=5)
                    )
                )
        
        # Update context panel
        self.context_panel.controls = [
            ft.Text("📊 Thông tin ngữ cảnh", size=18, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            *context_items
        ]
        
        self.context_panel.update()
    
    async def add_recent_query(self, query: str):
        """Add query to recent queries list"""
        if len(self.chat_history) >= 5:
            self.chat_history.pop(0)
        
        self.chat_history.append(query)
        
        # Update recent queries UI
        recent_items = [ft.Text("📝 Câu hỏi gần đây", size=16, weight=ft.FontWeight.W_500)]
        
        for q in reversed(self.chat_history[-3:]):  # Show last 3
            recent_items.append(
                ft.TextButton(
                    content=ft.Text(q[:30] + "..." if len(q) > 30 else q, size=10),
                    on_click=lambda e, query=q: asyncio.create_task(
                        self.send_message_programmatically(query)
                    )
                )
            )
        
        self.recent_queries_list.controls = recent_items
        self.recent_queries_list.update()
    
    async def load_suggestions(self):
        """Load initial suggestions from API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base_url}/chat/suggestions") as response:
                    if response.status == 200:
                        data = await response.json()
                        # Process suggestions if needed
        except:
            pass  # Fail silently
    
    async def call_chat_api(self, message: str) -> Dict:
        """Enhanced API call with better error handling"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.api_base_url}/chat/query",
                    json={
                        "query": message,
                        "user_id": "flet_user",
                        "context": {"interface": "flet", "voice_enabled": self.voice_enabled}
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"API Error {response.status}: {error_text}")
            except asyncio.TimeoutError:
                raise Exception("Request timeout - please try again")
            except aiohttp.ClientError as e:
                raise Exception(f"Connection error: {str(e)}")
    
    async def send_message_programmatically(self, message: str):
        """Send message programmatically from buttons/suggestions"""
        self.input_field.value = message
        self.input_field.update()
        await self.send_message()

# Additional utility functions and configurations

class VoiceCommands:
    """Voice command processing for hands-free operation"""
    
    COMMANDS = {
        "tìm": "search",
        "tìm kiếm": "search", 
        "mở": "open",
        "hiển thị": "show",
        "báo cáo": "report",
        "quy trình": "procedure",
        "hướng dẫn": "guide"
    }
    
    @staticmethod
    def process_voice_command(text: str) -> str:
        """Process voice command and convert to proper query"""
        text_lower = text.lower()
        
        # Check for specific commands
        for cmd, action in VoiceCommands.COMMANDS.items():
            if cmd in text_lower:
                return f"{action}: {text}"
        
        return text

class DataExport:
    """Export chat history and data"""
    
    @staticmethod
    async def export_chat_history(chat_history: List[dict], format: str = "json"):
        """Export chat history to file"""
        import json
        from datetime import datetime
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "chat_count": len(chat_history),
            "messages": chat_history
        }
        
        if format == "json":
            return json.dumps(export_data, ensure_ascii=False, indent=2)
        elif format == "txt":
            lines = [f"Chat History - Exported: {export_data['exported_at']}\n"]
            for msg in chat_history:
                lines.append(f"[{msg.get('timestamp', 'N/A')}] {msg.get('sender', 'Unknown')}: {msg.get('message', '')}\n")
            return "\n".join(lines)

# Configuration and settings
class AppConfig:
    """Application configuration"""
    
    # API Settings
    API_BASE_URL = "http://localhost:8000/api/v1"
    API_TIMEOUT = 30
    
    # Voice Settings
    VOICE_LANGUAGE = "vi-VN"
    TTS_RATE = 150
    VOICE_TIMEOUT = 10
    
    # UI Settings
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    CHAT_HISTORY_LIMIT = 100
    RECENT_QUERIES_LIMIT = 5
    
    # Colors
    Colors = {
        "primary": ft.Colors.BLUE,
        "secondary": ft.Colors.GREEN,
        "accent": ft.Colors.ORANGE,
        "error": ft.Colors.RED,
        "success": ft.Colors.GREEN,
        "warning": ft.Colors.ORANGE,
        "info": ft.Colors.BLUE
    }

# Main application entry point
def main():
    """Enhanced main function with configuration"""
    import os
    import sys
    import logging
    os.makedirs('logs', exist_ok=True)
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/flet_app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("WarehouseChatbot")
    logger.info("Starting Warehouse Chatbot Flet Application")
    
    try:
        # Create and run chatbot
        chatbot = AdvancedWarehouseChatbot()
        
        # Configure Flet app
        ft.app(
            target=chatbot.main,
            port=8080,
            view=ft.AppView.WEB_BROWSER,
            assets_dir="assets",  # For storing images, sounds, etc.
            upload_dir="uploads"  # For file uploads if needed
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()