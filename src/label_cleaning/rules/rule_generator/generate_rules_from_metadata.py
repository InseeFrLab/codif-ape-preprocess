import json
import os

from jinja2 import Environment, FileSystemLoader


def load_template(name: str):
    env = Environment(loader=FileSystemLoader("templates"))
    return env.get_template(name)


def snake_case(name):
    return name.lower().replace(" ", "_")


def generate_rule_files(metadata_path, output_dir="rules/naf_2025"):
    RULE_TEMPLATE = load_template("update_rule_template.j2")
    ADD_ROW_RULE_TEMPLATE = load_template("add_row_rule_template.j2")

    os.makedirs(output_dir, exist_ok=True)
    with open(metadata_path, "r", encoding="utf-8") as f:
        rules = json.load(f)

    for rule in rules:
        name = snake_case(rule["name"])
        content = RULE_TEMPLATE.render(**rule, name=name)
        if "new_row" in rule:
            content = ADD_ROW_RULE_TEMPLATE.render(**rule, name=name)
        else:
            content = RULE_TEMPLATE.render(**rule, name=name)

        with open(
            os.path.join(output_dir, f"rule_{name}.py"), "w", encoding="utf-8"
        ) as f:
            f.write(content)

    print("✅ Rule functions generated.")


if __name__ == "__main__":
    generate_rule_files(
        metadata_path="./metadata/rules_metadata.yaml",
        output_dir="./src/rules/naf_2025",
    )
