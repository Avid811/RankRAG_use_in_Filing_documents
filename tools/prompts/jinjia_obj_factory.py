from jinja2 import Template
import json


def get_jinjia_obj(jinjia_path:str):


    # 1. 读取模板文件
    with open(jinjia_path, 'r', encoding='utf-8') as f:
        template_content = f.read()

    # 2. 创建 Jinja2 模板对象
    template = Template(template_content)

    return template


if __name__ == "__main__":
    topm_template = get_jinjia_obj('rerank_model.j2')
    data = {
        'top_m':['1','2','3'],
        'part_article_content':"nihao"
    }
    print(topm_template.render(**data))


    """
# 工作内容
按照我所给的top_m文档与目标文段的语义关联性进行重排序，按照重要性降序输出

<top_m文档>['1', '2', '3']</top_m文档>
<目标文段>nihao</目标文段>
    """