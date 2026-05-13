from typing import List, Callable, Tuple, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
import os
import inspect
import re
import ast
import platform
from string import Template
from prompt_template import planner_system_prompt_template, executor_system_prompt_template


class PlanAndExecuteAgent:
    def __init__(self, tools: List[Callable], model: str, project_directory: str, max_iterations: int = 20):
        self.tools = { func.__name__: func for func in tools}
        self.model = model
        self.project_directory = project_directory
        self.max_iterations = max_iterations
        self.client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=PlanAndExecuteAgent.get_api_key(),
        )

    def run(self, user_input: str):
        plan = self.create_plan(user_input)
        print(f"\n Plan：\n{plan}")

        messages = [
            {"role": "system", "content": self.render_system_prompt(executor_system_prompt_template, plan=plan)},
            {"role": "user", "content": user_input}
        ]

        for _ in range(self.max_iterations):
            # 请求模型
            content = self.call_model(messages)

            # 检测 Current Step
            current_step = self.extract_tag(content, "current_step")
            if current_step:
                print(f"\n Current Step: {current_step}")

            # 检测 Thought
            thought = self.extract_tag(content, "thought")
            if thought:
                print(f"\n Thought: {thought}")
            
            # 检测是否输出 Final Answer，如果是，直接返回
            final_answer = self.extract_tag(content, "final_answer")
            if final_answer:
                return final_answer
            
            # 检测 Action
            action = self.extract_tag(content, "action")
            if not action:
                raise RuntimeError("模型未输出 <action>")
            tool_name, args, kwargs = self.parse_action(action)
            if tool_name not in self.tools:
                observation = f"工具不存在：{tool_name}。可用工具：{', '.join(self.tools.keys())}"
                messages.append({"role": "user", "content": f"<observation>{observation}</observation>"})
                continue

            formatted_args = ", ".join([repr(arg) for arg in args] + [f"{key}={value!r}" for key, value in kwargs.items()])
            print(f"\n Action: {tool_name}({formatted_args})")
            
            # 只有终端命令才需要询问用户，其他工具继续执行
            should_continue = input(f"请求调用Tool-{tool_name}\n是否继续？（Y/N）") \
                if tool_name in ["run_terminal_command", "write_to_file"] else "y"
            if should_continue.lower() != 'y':
                print("\n操作已取消。")
                return "操作已被用户取消"
            
            try:
                observation = self.tools[tool_name](*args, **kwargs)
            except Exception as e:
                observation = f"工具执行错误：{str(e)}"
            
            print(f"\n Observation：{observation}")
            obs_msg = f"<observation>{observation}</observation>"
            messages.append({"role": "user", "content": obs_msg})

        raise RuntimeError(f"执行超过最大轮数限制：{self.max_iterations}")


    def create_plan(self, user_input: str) -> str:
        """先规划，再进入工具执行循环。"""
        messages = [
            {"role": "system", "content": self.render_system_prompt(planner_system_prompt_template)},
            {"role": "user", "content": user_input}
        ]
        content = self.call_model(messages)
        plan = self.extract_tag(content, "plan")
        if not plan:
            raise RuntimeError("模型未输出 <plan>")
        return plan.strip()
    

    def render_system_prompt(self, system_prompt_template: str, **extra_values: str) -> str:
        """渲染系统提示模板，替换变量"""
        tool_list = self.get_tool_list()
        file_list = ", ".join(
            os.path.abspath(os.path.join(self.project_directory, f))
            for f in os.listdir(self.project_directory)
        )

        values = {
            "operating_system": self.get_operating_system_name(),
            "tool_list": tool_list,
            "file_list": file_list,
            **extra_values,
        }

        return Template(system_prompt_template).substitute(values)


    def get_tool_list(self) -> str:
        """生成工具列表字符串，包含函数签名和简要说明"""
        tool_descriptions = []
        for func in self.tools.values():
            name = func.__name__
            signature = str(inspect.signature(func))
            doc = inspect.getdoc(func)
            tool_descriptions.append(f"- {name}{signature}: {doc}")
        
        return "\n".join(tool_descriptions)


    def get_operating_system_name(self):
        os_map = {
            "Darwin": "macOS",
            "Windows": "Windows",
            "Linux": "Linux"
        }
        return os_map.get(platform.system(), platform.system())


    def call_model(self, message):
        print("\n请求模型中 ··· \n")
        response = self.client.chat.completions.create(
            model = self.model,
            messages=message,
        )

        # 上下文记录
        content = response.choices[0].message.content
        message.append(
            {"role": "assistant", "content": content}
        )

        return content

    def extract_tag(self, content: str, tag_name: str) -> str | None:
        match = re.search(rf"<{tag_name}>(.*?)</{tag_name}>", content, re.DOTALL)
        return match.group(1).strip() if match else None

    def parse_action(self, code_str: str) -> Tuple[str, List[Any], Dict[str, Any]]:
        code_str = code_str.strip()

        try:
            expression = ast.parse(code_str, mode="eval").body
            if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
                raise ValueError("Invalid function call syntax")

            args = [ast.literal_eval(arg) for arg in expression.args]
            kwargs = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in expression.keywords
                if keyword.arg is not None
            }
            return expression.func.id, args, kwargs
        except Exception:
            return self._parse_action_fallback(code_str)

    def _parse_action_fallback(self, code_str: str) -> Tuple[str, List[Any], Dict[str, Any]]:
        match = re.match(r'(\w+)\((.*)\)', code_str, re.DOTALL)
        if not match:
            raise ValueError("Invalid function call syntax")

        func_name = match.group(1)
        args_str = match.group(2).strip()

        # 解析参数，特别处理包含多行内容的字符串
        args = []
        current_arg = ""
        in_string = False   # 当前是不是在引号（字符串）里面
        string_char = None      # 包围当前字符串的是单引号 ' 还是双引号 "
        paren_depth = 0     # 当前在几层括号里面？
        i = 0

        while i < len(args_str):
            char = args_str[i]

            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    string_char = char
                    current_arg += char
                elif char == "(":
                    paren_depth += 1
                    current_arg += char
                elif char ==")":
                    paren_depth -= 1
                    current_arg += char
                elif char == ',' and paren_depth == 0:
                    # 遇到顶层或逗号，结束当前参数
                    args.append(self._parse_single_arg(current_arg.strip()))
                    current_arg = ""
                else:
                    current_arg += char
            else:
                current_arg += char
                if char == string_char and (i==0 or args_str[i-1] != '\\'):
                    in_string = False
                    string_char = None

            i += 1

        # 处理最后一个参数
        if current_arg.strip():
            args.append(self._parse_single_arg(current_arg.strip()))

        return func_name, args, {}

    def _parse_single_arg(self, arg_str: str):
        """解析单个参数"""
        arg_str = arg_str.strip()

        # 如果是字符串字面量
        if (arg_str.startswith('"') and arg_str.endswith('"')) or \
           (arg_str.startswith("'") and arg_str.endswith("'")):
            # 移除外层引号并处理转义字符
            inner_str = arg_str[1:-1]
            # 处理常见的转义字符
            inner_str = inner_str.replace('\\"', '"').replace("\\'", "'")
            inner_str = inner_str.replace('\\n', '\n').replace('\\t', '\t')
            inner_str = inner_str.replace('\\r', '\r').replace('\\\\', '\\')

            return inner_str
        
        # 尝试使用ast.literal_eval 解析其他类型
        try:
            return ast.literal_eval(arg_str)
        except (SyntaxError, ValueError):
            # 如果解析失败，则返回原始字符串
            return arg_str

    @staticmethod
    def get_api_key() -> str:
        """Load the API key from an environment variable."""
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("未找到 OPENROUTER_API_KEY 环境变量，请在 .env 文件中设置。")
        return api_key


# Backward-compatible alias for older code paths.
ReActAgent = PlanAndExecuteAgent
    
