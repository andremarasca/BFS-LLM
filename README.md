# BFS-LLM

Sistema de expansão de árvores conceituais usando BFS (Breadth-First Search) e LLMs.

## Início Rápido

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Configurar API key
export OPENAI_API_KEY='sua-chave-aqui'

# 3. Executar
python main.py
```

## Instalação

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar API Key (OpenAI)

Criar arquivo `.env` baseado em `.env.example`:

```bash
cp .env.example .env
```

Editar `.env` e adicionar sua chave:

```
OPENAI_API_KEY=sk-your-key-here
```

**Windows (PowerShell/CMD):**
```cmd
set OPENAI_API_KEY=sk-your-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY='sk-your-key-here'
```

## Execução

### Modo 1: Script direto

```bash
python main.py
```

### Modo 2: Scripts auxiliares

**Windows:**
```cmd
run.bat
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

## Configuração

Editar `config.yaml` para ajustar parâmetros:

### Execução

```yaml
execucao:
  dry_run: false  # true = simula sem chamar LLM
```

### BFS

```yaml
bfs:
  max_profundidade: 5      # níveis máximos (-1 = ilimitado)
  max_nos_por_nivel: -1    # nós por nível (-1 = ilimitado)
```

### LLM

```yaml
llm_modos:
  usar_local: false  # true = usa modelo local Qwen
  caminho_modelo_local: "./models/Qwen2.5-14B-Instruct-Q4_K_M.gguf"

llm_cliente:
  modelo: "gpt-4o-mini"
  temperatura: 0.1
  timeout_s: 60
```

### Retry

```yaml
retry:
  max_tentativas: 3
  timeout_entre_tentativas: 2
  continuar_em_falha: true
```

### Validação

```yaml
validacao:
  validar_schema: true
  permitir_duplicados: false
```

### Logging

```yaml
logging:
  nivel: "INFO"  # DEBUG, INFO, WARNING, ERROR
  salvar_em_arquivo: true
  arquivo_log: "./bfs_motor.log"
```

## Arquivos de Entrada

### `base_prompt.json`

Contém:
- **prompt**: Template base para o LLM
- **general_instructions**: Instruções gerais
- **prohibitions**: Restrições
- **concept_tree**: Árvore inicial
- **node_to_expand**: Exemplo de nó
- **node_json_schema**: Schema do nó
- **expected_output**: Formato esperado

### `json_schema.json`

Define estrutura de validação para sub-conceitos retornados pelo LLM.

## Saída

### Arquivos gerados

- **`concept_tree_output.json`**: Árvore expandida final
- **`snapshots/`**: Snapshots por nível BFS
- **`llm_traffic/`**: Logs de requisições/respostas LLM
- **`bfs_motor.log`**: Log geral da execução

### Estrutura do snapshot

```
snapshots/
├── snapshot_level_000.json
├── snapshot_level_001.json
├── snapshot_level_002.json
└── ...
```

### Estrutura do traffic log

```
llm_traffic/
└── YYYYMMDD_HHMMSS/
    ├── 0001_Entrada.json
    ├── 0001_Saida.json
    ├── 0001_Conjunto.json
    ├── 0002_Entrada.json
    └── ...
```

## Testes

Executar suite de testes:

```bash
python test_bfs.py
```

Testes incluem:
- JsonValidator
- PartialPersistence
- RetryPolicy
- BFSTreeExpander

## Fluxo de Execução

1. **Carrega** `config.yaml`, `base_prompt.json`, `json_schema.json`
2. **Inicializa** cliente LLM (OpenAI ou Qwen local)
3. **Percorre** árvore com BFS
4. **Identifica** nós elegíveis (sub_concepts vazio, is_leaf_node não true)
5. **Expande** cada nó via LLM
6. **Valida** resposta contra JSON Schema
7. **Atualiza** nó com sub-conceitos ou marca como leaf
8. **Persiste** snapshot ao final de cada nível
9. **Repete** até profundidade máxima ou não haver nós elegíveis
10. **Salva** árvore final em `concept_tree_output.json`

## Determinismo

Sistema garante determinismo:
- Mesma entrada + mesmas configurações = mesma saída
- Temperatura baixa (0.1) para respostas consistentes
- BFS sequencial por nível

## Tolerância a Falhas

- **Retry automático**: Até N tentativas por nó
- **Persistência parcial**: Snapshots por nível
- **Logs completos**: Auditoria de cada chamada LLM
- **Continue on failure**: Opção de continuar mesmo se nó falhar

## Uso com Modelo Local

### 1. Baixar modelo Qwen

Baixar GGUF de https://huggingface.co/ e colocar em `./models/`

### 2. Configurar

```yaml
llm_modos:
  usar_local: true
  caminho_modelo_local: "./models/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
```

### 3. Executar

```bash
python main.py
```

## Troubleshooting

### Erro: OPENAI_API_KEY not set

**Solução:** Configurar variável de ambiente conforme seção "Configurar API Key"

### Erro: Model file not found

**Solução:** Verificar caminho do modelo local em `config.yaml`

### Erro: JSON Schema validation failed

**Solução:** LLM retornou formato inválido. Verificar logs em `llm_traffic/` e ajustar prompt em `base_prompt.json`

### Erro: Rate limit

**Solução:** Aumentar `delay_entre_requests` em `config.yaml`:

```yaml
performance:
  delay_entre_requests: 1.0  # segundos
```

## Performance

### Rate limiting

```yaml
performance:
  delay_entre_requests: 0.5  # segundos entre requests
```

### Profundidade controlada

```yaml
bfs:
  max_profundidade: 3  # limitar níveis
  max_nos_por_nivel: 10  # limitar nós por nível
```

## Arquitetura

```
BFS-LLM/
├── core/
│   ├── llm_base.py          # BaseLLM, TrafficLogger, LLMConfig
│   ├── llm_openai.py        # OpenAILLM, JsonExtractor
│   ├── llm_local_qwen.py    # LocalQwenLLM
│   ├── llm_bfs_motor.py     # BFSMotor, BFSTreeExpander, etc.
│   └── __init__.py
├── main.py                   # Ponto de entrada
├── test_bfs.py              # Suite de testes
├── config.yaml              # Configuração
├── base_prompt.json         # Prompt e árvore inicial
├── json_schema.json         # Schema de validação
└── requirements.txt         # Dependências
```

## Próximos Passos

1. Ajustar `base_prompt.json` com seu domínio específico
2. Configurar `config.yaml` conforme necessidade
3. Executar com dry_run: true para testar
4. Executar com dry_run: false para expansão real
5. Analisar resultados em `concept_tree_output.json`
6. Iterar ajustando prompt e configurações
