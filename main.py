import click
import os
from agent import PlanAndExecuteAgent
from tools import *


@click.command()
@click.argument('project_dir',
                type=click.Path(exists=True, file_okay=False, dir_okay=True),
                default='.')
def main(project_dir):
    project_dir = os.path.abspath(project_dir)

    tools = [read_file, write_to_file, run_terminal_command]
    agent = PlanAndExecuteAgent(tools=tools, model="deepseek-v4-flash", project_directory=project_dir)

    task = input("请输入任务：")
    final_answer = agent.run(task)

    print(f"\n Final Answer：{final_answer}")


if __name__ == "__main__":
    main()
