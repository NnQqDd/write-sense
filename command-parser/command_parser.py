import os
import openai # Vẫn cần nếu bạn muốn kiểm tra key hoặc dùng các chức năng khác
from typing import List, Optional
import dotenv

dotenv.load_dotenv()

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field 
from langchain_core.exceptions import OutputParserException

# --- Định nghĩa cấu trúc Output mong muốn bằng Pydantic ---
# LangChain sẽ hướng dẫn LLM trả về dữ liệu theo cấu trúc này.
class CommandSequence(BaseModel):
    """Represents the sequence of commands extracted from a user request."""
    commands: List[str] = Field(
        ..., # Dấu ... nghĩa là trường này bắt buộc
        description="A list of valid command strings derived from the user request, in the exact order they should be executed. Must only contain commands from the provided available list."
    )
    # Có thể thêm các trường khác nếu cần, ví dụ: lý do, tham số lệnh,...
    # reason: Optional[str] = Field(None, description="Brief explanation for the chosen command sequence, if helpful.")

# --- Lớp CommandParser sử dụng LangChain ---
class CommandParser:
    def __init__(self, command_tree: dict[str, list[str]], model_name: str = "gpt-4o-mini"):
        """
        Khởi tạo CommandParser với đồ thị lệnh và model LangChain OpenAI.

        Args:
            command_tree: Đồ thị lệnh dạng danh sách kề (dict).
            model_name: Tên mô hình OpenAI muốn sử dụng (mặc định: "gpt-4o-mini").
        """
        if not isinstance(command_tree, dict):
            raise ValueError("command_tree phải là một dictionary")

        # Kiểm tra API Key trước khi khởi tạo LLM
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Biến môi trường OPENAI_API_KEY chưa được đặt.")

        self.command_tree = command_tree
        self.model_name = model_name

        # Trích xuất và lưu trữ danh sách lệnh hợp lệ
        self.all_commands = set(command_tree.keys())
        for destinations in command_tree.values():
            self.all_commands.update(destinations)
        self.all_commands_list = sorted(list(self.all_commands))

        # Khởi tạo LLM của LangChain
        # Temperature thấp để kết quả nhất quán hơn khi cần output theo cấu trúc
        try:
            self.llm = ChatOpenAI(model=self.model_name, temperature=0.1, openai_api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"Không thể khởi tạo ChatOpenAI. Lỗi: {e}")

        # Tạo prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant specialized in translating natural language user requests into a sequence of valid computer commands.
Your goal is to understand the user's intent and map it to a sequence of commands from the provided list.
Respond ONLY with the structured JSON output conforming to the requested schema. Do NOT include any explanations or conversational text outside the JSON structure.
The sequence of commands in the output list MUST follow the order specified in the user's request.
Only use commands from the following list of available commands:
{available_commands}
If the user request is ambiguous, unclear, or cannot be mapped to the available commands, return an empty list for the 'commands' field."""),
            ("human", "Translate the following user request: \"{user_command}\"")
        ])

        # Liên kết LLM với cấu trúc Pydantic mong muốn
        # LangChain sẽ tự động thêm hướng dẫn vào prompt và parse output
        try:
            self.structured_llm = self.llm.with_structured_output(CommandSequence)
        except Exception as e:
            # Có thể xảy ra lỗi nếu model không hỗ trợ function calling/structured output tốt
             raise RuntimeError(f"Model '{self.model_name}' có thể không hỗ trợ structured output tốt hoặc có lỗi cấu hình. Lỗi: {e}")

        # Tạo chain
        self.chain = self.prompt | self.structured_llm

        print(f"CommandParser initialized with LangChain. Model: '{self.model_name}'.")
        print(f"All known commands: {self.all_commands_list}")

    def tokenize_command(self, natural_language_command: str) -> list[str]:
        """
        Sử dụng LangChain và structured output để dịch câu lệnh tự nhiên thành tok_lst.

        Args:
            natural_language_command: Câu lệnh đầu vào của người dùng.

        Returns:
            tok_lst: Danh sách các token lệnh hợp lệ được trích xuất.
                     Trả về list rỗng nếu có lỗi hoặc không trích xuất được.
        """
        tok_lst = []
        if not natural_language_command or not isinstance(natural_language_command, str):
            print("Error: Input command must be a non-empty string.")
            return tok_lst

        print(f"\n--- Calling LangChain Chain (Model: {self.model_name}) ---")
        print(f"Input Natural Language: '{natural_language_command}'")

        try:
            # === Gọi LangChain chain ===
            response: CommandSequence = self.chain.invoke({
                "user_command": natural_language_command,
                "available_commands": str(self.all_commands_list) # Truyền list dưới dạng string cho an toàn
            })

            print(f"Raw structured response from LangChain: {response}")

            # === Validate kết quả từ Pydantic object ===
            if response and response.commands is not None:
                 # Lọc lần cuối để đảm bảo chỉ trả về các lệnh thực sự hợp lệ
                 valid_tokens = []
                 unknown_tokens = []
                 for command in response.commands:
                     if command in self.all_commands:
                         valid_tokens.append(command)
                     else:
                         unknown_tokens.append(command)

                 if unknown_tokens:
                      print(f"Warning: LLM returned commands not in the known list: {unknown_tokens}. They will be ignored.")

                 tok_lst = valid_tokens
            else:
                 # Trường hợp LLM trả về cấu trúc nhưng list commands là None hoặc rỗng
                 print("LLM returned a valid structure but no commands were extracted.")
                 tok_lst = []


        except OutputParserException as e:
            # Lỗi này xảy ra khi LLM trả về output không tuân thủ schema Pydantic
            print(f"Error: LLM output did not conform to the expected structure. Error: {e}")
            tok_lst = []
        except openai.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
        except openai.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
        except openai.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
        except Exception as e:
            # Bắt các lỗi khác từ LangChain hoặc quá trình xử lý
            print(f"An unexpected error occurred during chain execution: {e}")
            tok_lst = []

        print(f"--- LangChain Chain Execution Finished ---")
        return tok_lst

if __name__ == "__main__":

# 1. Xây dựng đồ thị trạng thái ứng dụng (command_tree)
    command_tree_complex = {
        'start_program': ['run_program', 'exit_program'],
        'run_program': ['execute', 'exit_program'],
        'execute': ['exit_program'],
        'exit_program': []
    }

    # 2. Khởi tạo Parser với LangChain
    # (Đảm bảo biến môi trường OPENAI_API_KEY đã được đặt)
    try:
        parser = CommandParser(command_tree_complex) # Mặc định dùng gpt-3.5-turbo

        # 3. Tập kiểm thử
        test_commands = [
            "Start the program then execute it and finally exit",
            "Bắt đầu chương trình, chạy nó, thực thi rồi thoát.",
            "run program",
            "Thoát chương trình",
            "Tôi muốn dừng lại",
            "Làm gì đó không rõ ràng",
            "Start and stop",
            "Chỉ cần chạy thôi",
            "run, execute, exit"
        ]

        # 4. Sử dụng Parser
        for command_text in test_commands:
            print(f"\nProcessing user command: '{command_text}'")
            token_list = parser.tokenize_command(command_text)
            print(f"Resulting token list (tok_lst): {token_list}")
            print("-" * 40)

    except ValueError as e:
        print(f"Initialization Error: {e}")
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during setup or execution: {e}")