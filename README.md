# Preprocessing toolkit for training data applied to business classification

## Gestionnaire des règles métiers

rules/*.py
  ↓
@rule(...)  →  register_rule(...)  →  RULE_REGISTRY.append(...)
  ↓
loader.py (import tous les .py → déclenche l’enregistrement dans RULE_REGISTRY)
  ↓
main.py (filtre les règles depuis le registre → applique)
