"""
Script de avaliação para testar a qualidade do modelo ICD-10-PCS
Testa códigos de procedimentos com casos de teste manuais
"""

import sys
import os
import re
from pathlib import Path

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llms import GoogleGenAILLM
from src.models.agent_model import AgentManager
import logging

# Configurar logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Reduzir logs do httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

print("="*70)
print("AVALIAÇÃO DO MODELO ICD-10-PCS - CÓDIGOS DE PROCEDIMENTOS")
print("="*70)

# Casos de teste manualmente curados
TEST_CASES = [
    {
        "description": "Laparoscopic appendectomy",
        "expected_code": "0DTJ4ZZ",
        "notes": "Resection of appendix, percutaneous endoscopic"
    },
    {
        "description": "Open cholecystectomy",
        "expected_code": "0FT40ZZ",
        "notes": "Resection of gallbladder, open approach"
    },
    {
        "description": "Percutaneous coronary angioplasty of single coronary artery",
        "expected_code": "02703ZZ",
        "notes": "Dilation of coronary artery, percutaneous"
    },
    {
        "description": "Total knee replacement, right",
        "expected_code": "0SRC0JZ",
        "notes": "Replacement of right knee joint"
    },
    {
        "description": "Diagnostic bronchoscopy",
        "expected_code": "0BJ08ZZ",
        "notes": "Inspection of trachea, via natural or artificial opening endoscopic"
    },
    {
        "description": "Percutaneous liver biopsy",
        "expected_code": "0FB03ZX",
        "notes": "Excision of liver, percutaneous, diagnostic"
    },
    {
        "description": "Open inguinal hernia repair with mesh",
        "expected_code": "0YU60JZ",
        "notes": "Supplement abdominal wall with synthetic substitute, open"
    },
    {
        "description": "Laparoscopic hysterectomy",
        "expected_code": "0UT94ZZ",
        "notes": "Resection of uterus, percutaneous endoscopic"
    },
    {
        "description": "Right total hip arthroplasty",
        "expected_code": "0SR9019",
        "notes": "Replacement of right hip joint with metal on polyethylene"
    },
    {
        "description": "Colonoscopy with polypectomy",
        "expected_code": "0DBN8ZZ",
        "notes": "Excision of large intestine, via natural opening endoscopic"
    },
]

# Inicializar o modelo
print("\n[1/3] Inicializando o modelo...")
llm = GoogleGenAILLM(model_name="gemini-2.5-flash", temperature=0.0)
agent_manager = AgentManager(llm=llm)
print("✓ Modelo inicializado com sucesso")

def extract_pcs_codes(text: str) -> list:
    """
    Extrai códigos ICD-10-PCS de uma resposta de texto.
    Procura por padrões de 7 caracteres alfanuméricos.
    """
    if not text:
        return []
    
    # Padrão para ICD-10-PCS: 7 caracteres alfanuméricos, SEM pontos
    # Pode estar após "Code:", "**" ou outros marcadores
    pattern = r'\b[0-9A-Z]{7}\b'
    matches = re.findall(pattern, text)
    
    # Filtrar códigos que parecem válidos (começam com 0-9, não todos números)
    valid_codes = []
    for match in matches:
        # Deve começar com seção (0-9 ou B,C,D,F,G,H,X)
        if match[0] in '0123456789BCDFGHX':
            # Não deve ser só números (evitar timestamps, etc)
            if not match.isdigit():
                valid_codes.append(match)
    
    return valid_codes

def normalize_pcs_code(code: str) -> str:
    """Normaliza código PCS: remove espaços, converte uppercase."""
    return code.strip().upper().replace(' ', '')

def match_position(predicted: str, expected: str, position: int) -> bool:
    """Verifica se uma posição específica do código está correta."""
    if len(predicted) <= position or len(expected) <= position:
        return False
    return predicted[position] == expected[position]

print(f"\n[2/3] Iniciando avaliação com {len(TEST_CASES)} casos...")
print("-"*70)

results = []
exact_matches = 0
partial_matches = []

for idx, case in enumerate(TEST_CASES, 1):
    description = case["description"]
    expected = normalize_pcs_code(case["expected_code"])
    notes = case.get("notes", "")
    
    print(f"\n{idx}. Testando: {description}")
    
    try:
        # Chamar o agente
        result = agent_manager.chat(description, thread_id=None)
        
        if result.get("status") != "completed":
            print(f"   ✗ Status não completado: {result.get('status')}")
            results.append({
                'case': idx,
                'description': description,
                'expected': expected,
                'predicted': 'ERROR',
                'match': False,
                'notes': notes
            })
            continue
        
        answer = result.get("answer", "")
        
        # Extrair códigos
        predicted_codes = extract_pcs_codes(answer)
        
        if not predicted_codes:
            print(f"   ✗ Nenhum código PCS encontrado na resposta")
            print(f"   Resposta: {answer[:200]}...")
            predicted = "NONE"
        else:
            predicted = normalize_pcs_code(predicted_codes[0])
            print(f"   Códigos encontrados: {predicted_codes}")
        
        # Verificar match
        exact_match = (predicted == expected)
        
        if exact_match:
            print(f"   ✓ CORRETO: {predicted}")
            exact_matches += 1
        else:
            # Análise por posição
            positions_correct = []
            if len(predicted) == 7 and len(expected) == 7:
                for pos in range(7):
                    if match_position(predicted, expected, pos):
                        positions_correct.append(pos + 1)
                
                positions_str = ','.join(map(str, positions_correct)) if positions_correct else 'nenhuma'
                print(f"   ✗ INCORRETO: {predicted} (esperado: {expected})")
                print(f"   Posições corretas: {positions_str}/7")
                partial_matches.append(len(positions_correct))
            else:
                print(f"   ✗ INCORRETO: {predicted} (esperado: {expected})")
                print(f"   Comprimento inválido: {len(predicted)} (esperado: 7)")
                partial_matches.append(0)
        
        results.append({
            'case': idx,
            'description': description,
            'expected': expected,
            'predicted': predicted,
            'match': exact_match,
            'notes': notes,
            'all_codes': predicted_codes
        })
        
    except Exception as e:
        print(f"   ✗ Erro: {e}")
        results.append({
            'case': idx,
            'description': description,
            'expected': expected,
            'predicted': 'ERROR',
            'match': False,
            'notes': notes
        })

print("\n" + "="*70)
print("[3/3] RESULTADOS DA AVALIAÇÃO")
print("="*70)

# Calcular métricas
total_cases = len(TEST_CASES)
accuracy = exact_matches / total_cases if total_cases > 0 else 0
avg_positions_correct = sum(partial_matches) / len(partial_matches) if partial_matches else 0

print(f"\n📊 Estatísticas Gerais:")
print(f"   Total de casos testados: {total_cases}")
print(f"   Matches exatos: {exact_matches}")
print(f"   Erros: {sum(1 for r in results if r['predicted'] == 'ERROR')}")
print(f"   Sem código: {sum(1 for r in results if r['predicted'] == 'NONE')}")

print(f"\n🎯 Métricas de Desempenho:")
print(f"   Exact Match Accuracy: {accuracy:.1%} ({exact_matches}/{total_cases})")
print(f"   Média de posições corretas (parciais): {avg_positions_correct:.1f}/7")

# Análise por posição
if partial_matches:
    print(f"\n📈 Análise de Precisão por Posição:")
    position_accuracy = [0] * 7
    for r in results:
        if len(r['predicted']) == 7 and len(r['expected']) == 7:
            for pos in range(7):
                if match_position(r['predicted'], r['expected'], pos):
                    position_accuracy[pos] += 1
    
    position_names = ['Section', 'Body System', 'Operation', 'Body Part', 'Approach', 'Device', 'Qualifier']
    valid_results = sum(1 for r in results if len(r['predicted']) == 7)
    
    for pos, (name, count) in enumerate(zip(position_names, position_accuracy), 1):
        accuracy_pos = count / valid_results if valid_results > 0 else 0
        print(f"   Pos {pos} ({name:12}): {accuracy_pos:.1%} ({count}/{valid_results})")

# Mostrar detalhes dos erros
errors = [r for r in results if not r['match']]
if errors:
    print(f"\n❌ Detalhes dos Erros ({len(errors)} casos):")
    print("-"*70)
    for r in errors:
        print(f"\n{r['case']}. {r['description']}")
        print(f"   Esperado:  {r['expected']} - {r['notes']}")
        print(f"   Obtido:    {r['predicted']}")

print("\n" + "="*70)
print("Avaliação concluída!")
print("="*70)

# Guardar resultados
output_file = Path(__file__).parent / "evaluation_pcs_results.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("RESULTADOS DA AVALIAÇÃO ICD-10-PCS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Total testado: {total_cases}\n")
    f.write(f"Exact Match Accuracy: {accuracy:.1%}\n")
    f.write(f"Média posições corretas: {avg_positions_correct:.1f}/7\n\n")
    f.write("Detalhes:\n")
    for r in results:
        status = "✓" if r['match'] else "✗"
        f.write(f"\n{status} {r['case']}. {r['description']}\n")
        f.write(f"   Esperado: {r['expected']} | Obtido: {r['predicted']}\n")
        f.write(f"   Notas: {r['notes']}\n")

print(f"\n💾 Resultados guardados em: {output_file}")

# Comparar com benchmarks anteriores
try:
    from benchmark_comparison import compare_benchmarks
    compare_benchmarks('icd10pcs', {
        'exact_match': accuracy,
        'avg_positions': avg_positions_correct / 7  # Normalizar para 0-1
    })
except Exception as e:
    print(f"\n⚠️  Não foi possível comparar com benchmarks: {e}")
