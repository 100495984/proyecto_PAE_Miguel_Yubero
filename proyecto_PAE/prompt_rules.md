# Reglas de dominio del sistema de clasificación PAE

Este documento recoge todas las instrucciones específicas de dominio que el sistema aplica sobre el LLM y sobre la lógica determinista para clasificar proyectos aeroespaciales en la taxonomía. Están organizadas por la fase del pipeline en que se aplican.

---

## 1. Prompt de perfil del proyecto (`_build_project_profile_prompt`)

Instrucciones que se envían al LLM para que extraiga un perfil técnico del proyecto antes de evaluar familias y categorías.

| Regla | Propósito |
|---|---|
| Distinguir la fuente del peligro del objetivo técnico real del trabajo. | Evita que un proyecto de protección frente a explosión de motor se clasifique como proyecto de propulsión. |
| Si el proyecto protege partes del avión de explosión, metralla, impacto o calor, no marcar propulsión como foco principal a menos que el motor sea el objetivo directo de diseño. | Caso concreto del punto anterior, muy frecuente en proyectos de blindaje estructural. |
| `evidence_phrases` debe contener frases copiadas o parafraseadas directamente del texto del proyecto. | Fuerza que las evidencias sean textuales y no inventadas. |
| `excluded_themes` debe nombrar dominios que aparecen solo como contexto y no deben dominar la clasificación. | Permite marcar explícitamente qué temas deben ignorarse como señal principal. |

---

## 2. Prompt de evaluación de familias (`_build_family_eval_prompt`)

Instrucciones que se envían al LLM para puntuar la relevancia de cada familia de la taxonomía (nivel raíz).

| Regla | Propósito |
|---|---|
| Juzgar SOLO esta familia, no otras. | Mantiene el foco en la evaluación individual. |
| Usar el foco real del proyecto, no solo palabras clave superficiales. | Evita falsos positivos por coincidencia léxica. |
| Si un dominio aparece solo como fuente de un peligro o contexto operativo, puntuar 0 o 1. | Reitera la separación hazard/target a nivel de evaluación de familias. |
| No puntuar familias paraguas (conceptos futuros, tecnologías disruptivas, nuevos materiales genéricos) por encima de 1 cuando una familia más específica explica mejor el proyecto. | Evita que categorías genéricas de innovación absorban proyectos con contenido técnico concreto. |
| Las aplicaciones futuras potenciales no cuentan como evidencia. | Evita clasificaciones especulativas. |
| La justificación debe ser concreta y específica del proyecto. | Obliga a citar evidencia real. |

---

## 3. Prompt de evaluación de categorías (`_build_category_eval_prompt`)

Instrucciones para que el LLM puntúe cada categoría individual dentro de una familia seleccionada.

| Regla | Propósito |
|---|---|
| Preferir el objetivo técnico del trabajo sobre la fuente del peligro. | Misma regla que en el perfil, reforzada aquí. |
| Si la categoría trata principalmente de propulsión, acústica, factores humanos u otro dominio lateral que no se diseña ni estudia directamente, puntuar 0 o 1. | Evita que categorías de dominio adyacente contaminen la clasificación. |
| Si el proyecto trata de escudos, fibras, composites, integración estructural, resistencia al impacto o fabricación de partes de avión, las categorías de estructuras/materiales/fabricación encajan mejor que las de propulsión. | Regla anti-confusión estructural/propulsión, muy frecuente en proyectos de blindaje. |
| Las mejoras futuras potenciales, beneficios especulativos o aplicaciones hipotéticas no cuentan. Si la categoría encaja solo porque el proyecto podría evolucionar en esa dirección, puntuar 0. | Evita clasificaciones prospectivas. |
| No usar categorías paraguas de innovación/nuevos materiales cuando una categoría más específica de estructuras/materiales/fabricación ya recoge la misma evidencia. | Mantiene la especificidad frente a lo genérico. |
| Las categorías de materiales inteligentes requieren evidencia explícita de detección, actuación, adaptabilidad o inteligencia embebida en el texto. | Evita que "sensor" o "composites" activen por inercia la categoría smart. |
| Las categorías de ruido o acústica requieren objetivos explícitos de acústica, sonido, ruido, vibración o medición. | Barrera estricta para evitar confusión entre ruido físico y otros contextos. |
| Las categorías de pruebas o validación requieren trabajo explícito de testing, medidas, caracterización, experimentos o validación. | Evita que proyectos que mencionan "verificar" tangencialmente se clasifiquen en testing. |
| Las categorías de seguridad requieren medidas explícitas de seguridad de aeronave/pasajeros/tripulación, no solo protección frente a daño físico. | Distingue "security" de "protection". |
| La justificación debe citar evidencia concreta del proyecto. | Obliga a citar el texto. |

---

## 4. Prompt de justificación final (`_build_reason_prompt`)

Instrucciones para que el LLM redacte la frase final de justificación de cada categoría seleccionada.

| Regla | Propósito |
|---|---|
| Mencionar detalles concretos del proyecto. | Justificaciones no genéricas. |
| Evitar frases genéricas como "falls under" o "belongs to". | Mejora la calidad redaccional. |
| No inventar actividades de testing, detección, acústica u otras que no sean explícitas en el texto del proyecto. | Evita alucinaciones del LLM. |
| Si el encaje es secundario y no central, hacerlo explícito en la frase. | Calibra correctamente el grado de relevancia. |
| No decir que el proyecto realiza testing, validación o fabricación avanzada a menos que el texto lo diga directamente. Si la categoría es solo un encaje metodológico secundario, decir que "apoya" o "se alinea con" el trabajo en lugar de que lo ejecuta. | Regla anti-sobreatribución de actividades. |

---

## 5. Prompt final de shortlist (`_build_final_shortlist_prompt`)

Instrucciones que se envían al LLM en la llamada principal de clasificación, donde elige categorías de un shortlist determinista.

| Regla | Propósito |
|---|---|
| Usar SOLO las categorías candidatas del shortlist. | Evita que el LLM invente categorías fuera del conjunto preseleccionado. |
| Ordenar de mayor a menor relevancia. | Garantiza ranking correcto. |
| Preferir el objetivo técnico real del trabajo sobre palabras de contexto. | Refuerzo general de la regla hazard/target. |
| **No confundir `ATM` con `atmospheric`.** | Regla explícita de desambiguación léxica muy frecuente. |
| **No tratar `shielding engine noise` como un escudo físico estructural.** | Distingue blindaje de ruido (acústico) de blindaje físico (estructural). |
| Las categorías de ruido interno requieren contexto de cabina, cabina de mando, pasajeros o ruido interior. | Barrera para "internal noise" frente a ruido externo o aeroacústica. |
| Las categorías de CFD requieren evidencia explícita de simulación computacional o numérica. | Evita que proyectos de túnel de viento se clasifiquen en CFD. |
| Las categorías de fabricación deben mantenerse secundarias a menos que el proyecto estudie claramente procesos de fabricación o métodos de producción. | Evita que "se fabricó un prototipo" eleve fabricación a categoría principal. |
| En modo weak-fit: elegir los proxies más cercanos de la taxonomía y dejarlo claro en el motivo. | Instrucción específica para proyectos que no encajan en ningún tema técnico fuerte. |
| En modo normal: elegir las coincidencias técnicas más fuertes; preferir menos categorías sobre opciones débiles o especulativas. | Instrucción de calidad para el caso estándar. |

---

## 6. Reglas de dominio por familia (`_apply_family_domain_rules`)

Reglas deterministas que sobreescriben la evaluación del LLM sobre familias enteras antes de que se seleccionen categorías. Se aplican en función de las señales detectadas en el texto del proyecto.

| Familia | Condición de bloqueo/fuerza | Motivo |
|---|---|---|
| Cualquier familia | Proyecto meta-no-técnico (impacto de programa, metodología sin tema aeroespacial) | Ninguna familia tiene encaje fuerte; se devuelven proxies metodológicos. |
| **1B0** Aerostructuras y materiales | Se detectan escudo/protección, fibras/composites Y diseño/fabricación | Se fuerza relevancia ≥ 2. Combinación que señala claramente este dominio. |
| **1C0** Propulsión | No se detecta propulsión como objetivo técnico directo | Se fuerza a 0. Regla ATI: propulsión solo como contexto de peligro no cuenta. |
| **1E0** Dinámica de vuelo | No hay trabajo de dinámica de vuelo ni análisis de fallos | Se fuerza a 0. |
| **1F0** Métodos y herramientas | No se detectan herramientas IT, ciclo de vida o ingeniería digital | Se fuerza a 0. |
| **1J0** Conceptos innovadores | No hay evidencia de conceptos o escenarios más allá del trabajo técnico central | Se fuerza a 0. Innovación genérica no es suficiente. |
| **1L0** Industria 4.0 | No hay evidencia de proceso digital | Se fuerza a 0. |
| **1A0** Aerodinámica | No hay trabajo explícito de aerodinámica o física de vuelo | Se fuerza a 0. |
| **1D0** Sistemas/aviónica | No hay desarrollo explícito de sistemas avionicos o equipos | Se fuerza a 0. |
| **1G0** ATM | No hay temas explícitos de gestión del tráfico aéreo | Se fuerza a 0. |
| **1H0** Aeropuerto | No hay temas relacionados con aeropuertos | Se fuerza a 0. |
| **1I0** Factores humanos | No hay temas de factores humanos o tripulación/pasajeros | Se fuerza a 0. |
| **1K0** UAS | No hay temas de sistemas no tripulados | Se fuerza a 0. |

---

## 7. Reglas de pre-chequeo por categoría (`_precheck_category_domain_rules`)

Reglas que bloquean categorías concretas antes de que se evalúen, en base a señales del proyecto. Si una categoría es bloqueada, se excluye del shortlist con relevancia = 0.

### Reglas por código de categoría

| Código | Señal requerida | Motivo |
|---|---|---|
| **1A1, 1A10** | `cfd_explicit` | CFD y acústica computacional requieren evidencia explícita de simulación de flujo numérico. |
| **1A2** | `unsteady_explicit` | Aerodinámica no estacionaria requiere evidencia de régimen transitorio o dinámico. |
| **1A5** | `high_lift` | Dispositivos de alta sustentación requieren evidencia de slats, flaps o configuración de despegue/aterrizaje. |
| **1A6** | `wing_design_terms` | Categoría de diseño de ala requiere evidencia de ala, carenado, góndola, fuselaje o empenaje. |
| **1A8** | `wind_tunnel` | Categoría de tecnología de túnel de viento requiere evidencia explícita de ensayo en túnel. |
| **1A9** | `wind_tunnel` AND `instrumentation` | Técnicas de medida en túnel requieren además evidencia de instrumentación. |
| **1A11** | `external_noise_context` | Predicción de ruido externo requiere evidencia aeroacústica o de ruido de motor externo. |
| **1B6** | `aeroelastic_explicit` | Aeroelasticidad requiere evidencia de flutter, respuesta a ráfagas o deformación estructural en flujo. |
| **1B7** | `vibration_explicit` OR `acoustics` | Pandeo/vibración/acústica requieren evidencia explícita de vibración, pandeo o acústica. |
| **1C10** | `bench_ground_test` | Banco de pruebas de propulsión requiere evidencia de banco, plataforma o ensayo en tierra. |
| **1C11** | NOT `out_of_domain_project` | Health monitoring de motor bloqueado en proyectos fuera del dominio aeroespacial. |
| **1C15** | `electrical_systems_core` | Potencia eléctrica requiere evidencia explícita de sistemas eléctricos o distribución de potencia. |
| **1D5, 1D10** | `avionics_systems` OR `electrical_systems_core` OR `systems_simulation_core` | Aviónica y electrónica a bordo requieren evidencia explícita de sistemas de a bordo. |
| **1D14** | `smart` | Mantenimiento inteligente requiere evidencia de materiales inteligentes o mantenimiento predictivo. |
| **1D23** | `fuel_systems_core` | Sistemas de combustible requieren evidencia explícita de sistemas de combustible. |
| **1F4** | `bench_ground_test` OR `wind_tunnel` OR (`testing_validation` AND `propulsion_core`) | Vuelos/ensayos en tierra requieren evidencia explícita de campaña, banco o ensayo de propulsión. |
| **1F22** | `meta_methodology` OR `study_assessment_core` | Métodos de investigación operacional requieren evidencia de estudio, trade-off o metodología. |
| **1F24** | `flight_dynamics` | Rendimiento de aeronave requiere evidencia de rendimiento, estabilidad, controlabilidad o mecánica de vuelo. |
| **1F25** | `airport` | Rendimiento de aeropuerto requiere evidencia explícita de aeropuerto. |
| **1F27** | `systems_simulation_core` | Modelos numéricos requieren evidencia explícita de entorno de simulación o modelos numéricos. |
| **1F32, 1F33** | `demonstrator_platform` AND (`bench_ground_test` OR `testing_validation`) | Validación a gran escala requiere evidencia de demostradores más actividad real de validación o ensayo. |
| **1G8, 1G9** | `airport` | Operaciones de aeropuerto requieren evidencia explícita de aeropuerto o tráfico aeroportuario. |
| **1I4** | `explicit_training` | Selección y formación requieren evidencia explícita de training o selección. |

### Regla especial: proyectos ATM/navegación sin trabajo físico aeroespacial

Si el proyecto es un proyecto central de navegación/ATM (`atm_navigation_core`) y **no tiene** evidencia de trabajo físico aeroespacial (`physical_aero_project`), se bloquean automáticamente las categorías: `1A1`, `1A2`, `1A5`, `1A6`, `1A7`, `1A8`, `1A9`, `1A10`, `1A11`, `1B10`, `1B11`, `1B12`, `1B13`.

**Motivo:** Los proyectos de navegación o procedimientos de vuelo no deben clasificarse en categorías de física aerodinámica o de ensayos acústicos si no hay evidencia real de ese trabajo.

### Reglas por etiqueta de categoría (tags)

Adicionalmente, las categorías se etiquetan automáticamente y se bloquean si no hay la señal correspondiente:

| Etiqueta | Señal requerida |
|---|---|
| `acoustics` | `acoustics` — el proyecto menciona objetivos acústicos, de ruido o sonido |
| `smart` | `smart` — el proyecto menciona materiales inteligentes, sensado, actuación o adaptabilidad |
| `security` | `security` — el proyecto menciona medidas de seguridad de tripulación/pasajeros |
| `testing` | `testing_validation` — el proyecto menciona ensayos, experimentos o medidas |
| `flight_dynamics` / familia 1E | `flight_dynamics` OR `failure_analysis` |
| `failure_analysis` / `hazard_analysis` | `failure_analysis` |
| `digital_process` | `digital_process` |
| `innovation_umbrella` | `innovation_concepts` — solo si el proyecto apunta explícitamente a conceptos o escenarios |
| `propulsion` | `propulsion_core` — bloqueado cuando la propulsión es solo contexto de peligro |
| `human` | `human` |
| `airport` | `airport` |
| `atm` | `atm` |
| `uas` | `uas` |
| `metallic` | `metallic_materials` — bloqueado si las evidencias apuntan a fibras/no metálicos |
| `nonmetal` | `nonmetal_materials` |
| `composite` | `composite_materials` OR `nonmetal_materials` |

---

## 8. Reglas de ajuste post-LLM por categoría (`_apply_category_domain_rules`)

Reglas que se aplican **después** de que el LLM haya evaluado una categoría, para corregir puntuaciones a la baja o al alza.

| Condición | Efecto | Motivo |
|---|---|---|
| Categoría no metálica (`nonmetal`) + señal `nonmetal_materials` en el proyecto | Fuerza relevancia ≥ 2 | Evidencia directa de materiales no metálicos justifica elevación automática. |
| Categoría compuesta (`composite`) + señal `composite_materials` o `nonmetal_materials` | Fuerza relevancia ≥ 2 | Materiales de fibra o compuestos son evidencia directa. |
| Categoría de diseño estructural (`structural_design`) + `structures_materials_core` + `design_work` | Fuerza relevancia ≥ 2 | Diseño estructural explícito justifica elevación automática. |
| Categoría de fabricación (`manufacturing`) + señal `manufacturing_process` | Fuerza relevancia ≥ 2 | Hay investigación real de procesos de fabricación. |
| Categoría de fabricación + solo `manufacture_explicit` (sin proceso) | Baja relevancia a máximo 1 (secundaria) | La fabricación es mencionada pero no es el objeto de investigación. |
| Categoría de fabricación + ninguna señal de fabricación | Bloqueo completo | No hay evidencia alguna de fabricación. |
| Categoría de testing + sin señal `testing_validation` | Bloqueo completo | No se mencionan ensayos, experimentos ni validación en el proyecto. |

---

## 9. Señales de proyecto y su definición resumida

El sistema detecta las siguientes señales booleanas en el texto del proyecto antes de aplicar cualquier regla. Cada señal activa o desactiva reglas específicas de las secciones anteriores.

| Señal | Qué detecta |
|---|---|
| `propulsion_core` | Términos de propulsión como objetivo técnico directo (turbina, combustor, empuje…), excluyendo "engine burst" o turbinas eólicas. |
| `aerodynamics_physics` | Términos aerodinámicos o trabajo en túnel de viento. |
| `wind_tunnel` | Mención explícita de túnel de viento. |
| `cfd_explicit` | CFD, simulación de flujo, solver aerodinámico. |
| `unsteady_explicit` | Aerodinámica no estacionaria, transitoria, flutter, dinámica. |
| `high_lift` | Slats, flaps, aerofrenos, configuraciones de alta sustentación. |
| `wing_design_terms` | Ala laminar, empenaje, carenado, fuselaje, góndola. |
| `acoustics` | Objetivos acústicos, de ruido o sonido. |
| `external_noise_context` | Ruido externo/aeroacústico (en combinación con aerodinámica o túnel). |
| `internal_noise_context` | Ruido interno, cabina, pasajeros, comodidad. |
| `fem_structural` | FEM, verificación estructural, cargas representativas. |
| `aeroelastic_explicit` | Flutter, respuesta aeroelástica, deformación estructural en flujo. |
| `vibration_explicit` | Vibración, pandeo, respuesta modal. |
| `structures_materials_core` | Combinación de escudo/composites/estructural + diseño/fabricación. |
| `shield_protection` | Contexto de protección física: metralla, explosión, resistencia al impacto. |
| `composite_materials` | Composites, matriz de fibra. |
| `nonmetal_materials` | Fibras, membranas, cerámicas, materiales orgánicos. |
| `metallic_materials` | Metales, aleaciones, aluminio, titanio. |
| `manufacturing_process` | Investigación de procesos de fabricación o producción. |
| `manufacture_explicit` | Mención explícita de fabricación sin ser el objeto de estudio. |
| `process_tooling_manufacturing` | Utillaje, metrología, tolerancias, industrialización. |
| `design_work` | Diseño, concepción, integración. |
| `testing_validation` | Ensayos, medidas, caracterización, validación, verificación. |
| `bench_ground_test` | Banco de pruebas, plataforma de ensayo, campaña en tierra. |
| `instrumentation` | Sondas, sensores, galgas, kulites, instrumentación de presión. |
| `systems_simulation_core` | Entornos de simulación, modelos numéricos, herramientas software. |
| `electrical_systems_core` | Sistemas eléctricos, distribución de potencia, arquitectura de a bordo. |
| `fuel_systems_core` | Sistemas de combustible. |
| `avionics_systems` | Aviónica, sistemas de a bordo, bus de comunicaciones. |
| `flight_dynamics` | Mecánica de vuelo, rendimiento, estabilidad y control, trayectoria. |
| `failure_analysis` | Análisis de fallos, bird strike, accidente, peligro ambiental. |
| `digital_process` | Herramientas IT, ciclo de vida, gemelo digital, ingeniería colaborativa. |
| `smart` | Materiales inteligentes, sensado, actuación, adaptabilidad. |
| `security` | Seguridad de pasajeros/tripulación, dispositivos de barrera. |
| `atm` | Gestión del tráfico aéreo, navegación, gestión de trayectorias. |
| `egnos_gnss` | EGNOS, GNSS, GPS, SBAS, LPV, RNAV, RNP. |
| `atm_navigation_core` | ATM + (GNSS/EGNOS o procedimientos de vuelo). |
| `flight_procedures` | Procedimientos IFR, perfiles de vuelo, rutas, despliegue operacional. |
| `certification_work` | Certificación, normativa, regulación, estándares de seguridad. |
| `human` | Factores humanos, carga de trabajo del piloto/controlador, entrenamiento. |
| `uas` | Sistemas no tripulados, drones, vuelo autónomo. |
| `airport` | Aeropuerto, pista, terminal, handling en tierra. |
| `innovation_concepts` | Conceptos de aeronave no convencional, propulsión eléctrica, análisis de escenarios. |
| `meta_methodology` | Evaluación del impacto de programas, metodología de leverage/driving effect. |
| `study_assessment_core` | Estudio de requisitos, trade-off, benchmark, análisis DAFO. |
| `demonstrator_platform` | Demostradores, plataformas de validación, entornos de simulación compartidos. |
| `physical_aero_project` | Al menos uno de: túnel, CFD, no estacionario, alta sustentación, ala, instrumentación, FEM, aeroelástico, vibración. |
| `strong_technical_topic` | Al menos un dominio técnico claro detectado. |
| `meta_nontechnical_project` | Proyecto de metodología/impacto sin tema técnico fuerte, o proyecto fuera del dominio aeroespacial. |
| `out_of_domain_project` | Proyecto de energía eólica, evaluación no destructiva u otro dominio industrial no aeroespacial. |
