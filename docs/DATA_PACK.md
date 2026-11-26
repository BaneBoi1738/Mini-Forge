# Mini-Forge data pack (fictional)

All content is fictional and created for a Neolytix take-home exercise. Patient names, dates of birth,
member IDs and claim numbers are fake test values. Treat them as if they were real PHI anyway.

- documents/  12 payer denial letters (DOC-001 … DOC-012), plain text
- sops/       the SOP library (mini "NeoSutra"); each file has front matter with a status
- config/agents.yaml    pre-approved agent catalogue
- config/policies.yaml  deny-by-default action policy, including which fields may be de-tokenized per tool
- labels.json           expected category, action and human-review flag per document, plus the PHI values
                        in each document (use them to prove no PHI reached the model)
