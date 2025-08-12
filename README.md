# Domain-specific data cleaning rules for business classification

## Gestionnaire des règles métiers

rules/*.py
  ↓
@rule(...)  →  register_rule(...)  →  RULE_REGISTRY.append(...)
  ↓
loader.py (import tous les .py → déclenche l’enregistrement dans RULE_REGISTRY)
  ↓
main.py (filtre les règles depuis le registre → applique)


Prochaine fois: schéma ==> raw data --> specific preprocessing before calibration --> apply business rules + modality injection --> output new raw data (cleaned with domain specific-knowledge)

business rule-based data cleansing

Data cleansing, the process of ensuring that the extracted data is ac-
curate, complete, and consistent—often involves labor-intensive processes that
require domain expertise, as detecting and correcting (or removing) corrupt or in-
accurate records from a dataset, is a crucial step in the data preparation pipeline.
High-quality data is essential for producing reliable and accurate insights in data-
driven research and applications. Traditionally, data cleansing has been handled
using a combination of rule-based methods, statistical techniques, and manual
interventions.