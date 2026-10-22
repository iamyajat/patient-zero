# Lecture outlines

## Lecture 1: Accountability gaps in agent swarms (50 min)
1. **What a swarm is (10 min).** Schroeder et al. (Science, 2026): persistent identities, coordination, adaptation,
   minimal oversight. Fabricated consensus as "counterfeit social proof". Contrast outward-facing swarms with the agent
   pipelines inside an organisation.
2. **How errors spread (15 min).** Live demo of the harness: a planted fact in the researcher's documents, and how far
   it travels in a chain compared with a hub. Introduce R̂ and why "R̂ < 1" is a criterion a regulator can understand.
3. **Diffuse causation and the "many hands" problem (15 min).** Who is responsible when no single agent erred badly:
   the developer of each agent, the integrator, the deployer? Gradient Institute/DISR (2026) failure classes. Cases:
   the AI Village fabricated contact list, and the Moltbook injection attempts.
4. **Discussion (10 min).** Is a message log enough to assign responsibility? What is missing from it (hidden beliefs,
   steganography)?

## Lecture 2: What counts as passing? (50 min)
1. **Audits and independence (10 min).** Raji et al. (2020, 2022): internal vs external audits, who pays the auditor,
   and the evidence that audits paid for by the audited firm are less accurate.
2. **Setting thresholds is an ethical choice (20 min).** False alarms vs missed contagion: who bears each cost? Why
   T3 fixes the false-alarm rate (users who are wrongly blocked) and then asks for recall. Risk tiers and δ.
   Exercise: move δ and R* and watch which systems pass (`audit/sensitivity.py` figures).
3. **Monitoring as power (10 min).** Reading every inter-agent message raises privacy, surveillance and
   power-concentration concerns (Chan et al. 2024). Who should see the logs, and for how long?
4. **From verdict to action (10 min).** Quarantine, incident reporting (EU AI Act Art. 73), re-certification on change.
   Preview of the assignment.
