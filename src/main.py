# # #
# EP 1
# Alunos:
# Átila da Veiga -- NUSP: 12491731
# Roberta de Souza Pereira -- NUSP: 13687794
# # #

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

FuncaoEscalar = Callable[[float], float]
Metodo = Literal["modificado", "somente_newton", "somente_bissecao"]


@dataclass(slots=True)
class Resultado:
    """Resultado da execução do metodo de Newton modificado.

    Attributes:
        x: Aproximacao final obtida para a raiz.
        fx: Valor de f(x) na aproximacao final.
        iteracoes: Numero total de iteracoes realizadas.
        convergiu: Indica se o criterio de parada foi atingido com sucesso.
        motivo_parada: Texto curto indicando o motivo da parada.
        contagem_newton: Quantidade de iteracoes que usaram o passo de Newton.
        contagem_bissecao: Quantidade de iteracoes que usaram dicotomia.
        historico_metodos: Lista com o metodo utilizado em cada iteracao.
    """

    x: float
    fx: float
    iteracoes: int
    convergiu: bool
    motivo_parada: str
    contagem_newton: int
    contagem_bissecao: int
    historico_metodos: list[str]


def zero_funcao(
    f: FuncaoEscalar,
    df: FuncaoEscalar,
    a: float,
    b: float,
    x0: float | None = None,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-10,
    maxit: int = 100,
    metodo: Metodo = "modificado",
    relatorio: bool = False,
) -> Resultado:
    """Calcula uma raiz aproximada usando o metodo de Newton modificado.

    O algoritmo combina o passo de Newton com dicotomia. Em cada iteracao, a
    escolha do metodo depende da permanencia da aproximacao dentro do intervalo
    atual que isola a raiz e da reducao aceitavel do tamanho do passo.

    Args:
        f: Funcao escalar cuja raiz se deseja aproximar.
        df: Derivada de f, usada nas iteracoes candidatas de Newton.
        a: Extremidade esquerda do intervalo inicial.
        b: Extremidade direita do intervalo inicial.
        x0: Aproximacao inicial opcional. Quando nao informada, e escolhido um
            dos extremos do intervalo inicial.
        atol: Tolerancia absoluta usada no criterio de parada.
        rtol: Tolerancia relativa usada no criterio de parada.
        maxit: Numero maximo de iteracoes permitidas.
        metodo: Estrategia de selecao do passo. "modificado" combina Newton e
            dicotomia conforme os criterios do enunciado (comportamento padrao).
            "somente_newton" usa exclusivamente o passo de Newton, sem recorrer
            a dicotomia mesmo que isso resulte em erro ou divergencia.
            "somente_bissecao" usa exclusivamente dicotomia; df nunca e chamada.
        relatorio: Se True, imprime um relatorio simples da execucao no stdout.

    Returns:
        Um objeto Resultado contendo a aproximacao final, o valor de f na
        aproximacao, o numero de iteracoes, o status de convergencia e o
        historico de metodos usados.

    Raises:
        TypeError: Se f ou df nao forem funcoes.
        ValueError: Se o intervalo for invalido, se as tolerancias forem
            negativas, se maxit nao for positivo, se nao houver troca de sinal
            no intervalo, se x0 nao for um dos extremos do intervalo ou se
            metodo nao for um dos valores aceitos.
    """
    # # TESTES INICIAIS DE VALIDADE DOS PARAMETROS MANUAIS
    if not callable(f) or not callable(df):
        raise TypeError("f e df devem ser funcoes funcoes.")
    if not (a < b):
        raise ValueError("O intervalo deve satisfazer a < b.")
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("As tolerancias atol e rtol devem ser nao negativas.")
    if maxit <= 0:
        raise ValueError("maxit deve ser positivo.")
    if metodo not in ("modificado", "somente_newton", "somente_bissecao"):
        raise ValueError(
            "metodo deve ser 'modificado', 'somente_newton' ou 'somente_bissecao'."
        )

    alpha = float(a)
    beta = float(b)
    f_alpha = float(f(alpha))
    fb = float(f(beta))

    if f_alpha * fb > 0.0:
        raise ValueError("O intervalo inicial deve ter troca de sinal: f(a)*f(b) < 0.")

    if x0 is None:
        xn = alpha if abs(f_alpha) <= abs(fb) else beta
    else:
        xn = float(x0)
        if xn < alpha or xn > beta:
            raise ValueError("x0 deve estar dentro do intervalo [a, b].")

    fxn = float(f(xn))

    # # INICIALIZACAO DE VARIAVEIS DE CONTROLE E HISTORICO DE METODOS
    iteracoes = 0
    convergiu = False
    contagem_newton = 0
    contagem_bissecao = 0
    historico_metodos: list[str] = []
    delta_xn: float | None = None

    for n in range(maxit):
        if metodo == "somente_bissecao":
            usar_newton = False
        elif metodo == "somente_newton":
            usar_newton = True
            dfxn = float(df(xn))
        else:  # Caso do metodo "modificado" (caso solicitado pelo EP)
            dfxn = float(df(xn))

            # # AVALIA CRITERIOS DE SELECAO DE METODO
            # Criterios 2 e 3 fazem referencia a numeracao dos critreios conforme o
            # enunciado do EP.
            criterio_2 = ((xn - alpha) * dfxn - fxn) * ((xn - beta) * dfxn - fxn) < 0
            if delta_xn is None:
                criterio_3 = True
            else:
                criterio_3 = 2.0 * abs(fxn) < abs(dfxn * delta_xn)
            usar_newton = criterio_2 and criterio_3

        if usar_newton:
            if np.isclose(dfxn, 0.0, atol=np.finfo(float).eps, rtol=0.0):
                raise ZeroDivisionError("Derivada nula em metodo de Newton.")
            xn1 = xn - fxn / dfxn  # metodo de Newton
            metodo_usado = "newton"
            contagem_newton += 1
        else:
            xn1 = 0.5 * (alpha + beta)
            metodo_usado = "dicotomia"
            contagem_bissecao += 1

        historico_metodos.append(metodo_usado)
        fxn1 = float(f(xn1))
        iteracoes = n + 1

        # Critério de parada (a)
        prox_zero = np.isclose(fxn1, 0.0, atol=atol, rtol=rtol)
        # Garantindo que esta proximo de zero para que o criterio (a) esteja correto.
        delta = xn1 - xn
        if abs(delta) < atol + rtol * abs(xn1) and prox_zero:
            xn = xn1
            fxn = fxn1
            convergiu = True
            motivo_parada = "tolerancia"
            break

        # Critério de parada (b)
        if iteracoes >= maxit:
            convergiu = False
            motivo_parada = "maxit"
            break

        # Critério de parada (c)
        if np.isclose(fxn1, 0.0, atol=atol, rtol=rtol):
            xn = xn1
            fxn = fxn1
            convergiu = True
            motivo_parada = "raiz_exata"
            break

        # Atualizacao do intervalo isolante pela troca de sinal.
        if f_alpha * fxn1 < 0.0:
            beta = xn1
        else:
            alpha = xn1
            f_alpha = fxn1

        delta = xn1 - xn

        xn = xn1
        fxn = fxn1
        delta_xn = delta

    # # CONSTRUCAO DO RESUMO DE EXECUCAO DO ALGORITMO
    resultado = Resultado(
        x=xn,
        fx=fxn,
        iteracoes=iteracoes,
        convergiu=convergiu,
        motivo_parada=motivo_parada,
        contagem_newton=contagem_newton,
        contagem_bissecao=contagem_bissecao,
        historico_metodos=historico_metodos,
    )

    if relatorio:
        print("Relatorio do metodo de Newton modificado")
        print(f"Intervalo inicial: [{a}, {b}]")
        print(f"ATOL={atol}, RTOL={rtol}, MAXIT={maxit}")
        print(f"Iteracoes realizadas: {resultado.iteracoes}")
        print(f"Passos de Newton: {resultado.contagem_newton}")
        print(f"Passos de dicotomia: {resultado.contagem_bissecao}")
        print(f"Convergiu: {resultado.convergiu}")
        print(f"Motivo da parada: {resultado.motivo_parada}")
        print(f"Aproximacao final: x={resultado.x}")
        print(f"Residuo final: |f(x)|={abs(resultado.fx)}")

    return resultado


# TAREFA 3.1: Queda de um corpo sob a acao da resistencia do ar
# # Implementacao
def encontrar_k_queda_corpo(
    M: float = 1.0,  # valor sugerido na tarefa
    g: float = 10.0,  # valor sugerido na tarefa
    v0: float = 3.0,  # valor sugerido na tarefa
    t: float = 2.0,
    v_alvo: float = 20.0,  # v(2) conforme tarefa
    a: float = 0.1,
    b: float = 5.0,
    relatorio: bool = False,
) -> Resultado:
    """Determina k tal que a velocidade de queda v(t) atinja v_alvo .

    Modela v(t) = (Mg - e^(-(t/M)k)(Mg - v0*k)) / k e usa zero_funcao para achar
    a raiz de f(k) = v(t) - v_alvo no intervalo [a, b].

    Args:
        M: Massa do corpo.
        g: Aceleracao da gravidade.
        v0: Velocidade inicial (em t=0).
        t: Instante em que a velocidade alvo deve ser atingida.
        v_alvo: Velocidade desejada no instante t.
        a: Extremidade esquerda do intervalo de busca para k.
        b: Extremidade direita do intervalo de busca para k.
        relatorio: Se True, imprime o relatorio de zero_funcao.

    Returns:
        O Resultado de zero_funcao aplicado a f(k) = v(t; k) - v_alvo.
    """
    c = t / M
    mg = M * g

    def f(k: float) -> float:
        n = mg - math.exp(-c * k) * (mg - v0 * k)
        return n / k - v_alvo

    def df(k: float) -> float:
        exp_ck = math.exp(-c * k)
        n = mg - exp_ck * (mg - v0 * k)
        dn = exp_ck * (c * (mg - v0 * k) + v0)
        return (dn * k - n) / (k * k)

    print("\n\nTAREFA 3.1:\n")
    return zero_funcao(f, df, a, b, relatorio=relatorio)


# TAREFA 3.2: Altura dos FIos de transmissao de eletrecidade
# # Implementacao
def encontrar_beta_catenaria(
    x_max: float = 10.0,  # Extremo do cabo onde a diferenca de altura e medida
    delta_alvo: float = 0.5,  # Objetivo
    a: float = 1,  # evitar ser zero
    b: float = 99999,
    relatorio: bool = False,
) -> Resultado:
    """Determina beta da catenaria y = alpha + beta*cosh(x/beta) .

    Como f(x_max) - f(0) nao depende de alpha, resolve-se
    func(beta) = beta*(cosh(x_max/beta) - 1) - delta_alvo = 0 via zero_funcao.

    Args:
        x_max: Extremidade do cabo onde a diferenca de altura e medida.
        delta_alvo: Diferenca de altura f(x_max) - f(0) desejada. 0.5 no exemplo.
        a: Extremidade esquerda do intervalo de busca para beta. Deve ser maior que zero.
        b: Extremidade direita do intervalo de busca para beta. Deve ser grande.
        relatorio: Se True, imprime o relatorio de zero_funcao.

    Returns:
        O Resultado de zero_funcao aplicado a func(beta).
    """

    def func(beta_: float) -> float:
        # lembrando, cosh(0/beta_)=1
        return beta_ * (math.cosh(x_max / beta_) - 1) - delta_alvo

    def d_func(beta_: float) -> float:
        r = x_max / beta_
        return math.cosh(r) - r * math.sinh(r) - 1.0

    print("\n\nTAREFA 3.2:\n")
    return zero_funcao(func, d_func, a, b, relatorio=relatorio)


# TAREFA 4: Formulas de quadratura de Gauss-Legendre
# Resultado
def avaliar_legendre(n, x):
    """Função que calcula, no ponto x, os valores do polinômio de
    Legendre de grau n e de sua derivada usando as fórmulas de recorrência
    (k + 1)P_k+1(x) = (2k + 1)xP_k(x) - kP_k-1(x)
    e
    P_k'(x) =k(xP_k - P_k-1)/(x**2 - 1.0)"""

    # Consistência para n:
    if not isinstance(n, int):
        raise TypeError("O grau do polinômio deve ser um inteiro.")
    if n < 0:
        raise ValueError("O grau do polinômio deve ser um inteiro não negativo.")

    # Consistência para x:
    if not isinstance(x, (int, float)):
        raise TypeError("O ponto x deve ser um número.")

    # Condição trivial para P_0(x)
    if n == 0:
        return 1.0, 0.0

    # Inicializa P_0(x) e P_1(x)
    p_k_menos_1 = 1.0  # Guarda P_0(x) = 1
    p_k = x  # Guarda P_1(x) = x

    # Laço para construir o valor até o grau n
    for k in range(1, n):
        # Aplica a fórmula de recorrência
        p_proximo = ((2.0 * k + 1.0) * x * p_k - k * p_k_menos_1) / (k + 1.0)

        # Atualiza as variáveis para a próxima iteração do laço
        p_k_menos_1 = p_k
        p_k = p_proximo

    # Quando o laço termina, p_k é o valor numérico de P_n(x) e
    # p_k_menos_1 é o valor numérico de P_{n-1}(x).

    # Calcula P_n'(x). Nos extremos x = ±1, utiliza as expressões
    # específicas para evitar a divisão por zero na fórmula geral.
    if x == 1.0:
        derivada = n * (n + 1.0) / 2.0
    elif x == -1.0:
        if n % 2 == 0:
            derivada = (-1.0) * n * (n + 1.0) / 2.0
        else:
            derivada = n * (n + 1.0) / 2.0
    else:
        derivada = n * (x * p_k - p_k_menos_1) / (x**2 - 1.0)

    return p_k, derivada


def calcula_valores_quadratura(grau: int, imprimir: bool = True):
    """Calcula as raízes, ou nós, não negativos para os polinômios de Legendre de grau N, para N=1,...,grau,
    e os respectivos pesos omega_j da Fórmula de Quadratura de Gauss-Legendre.
    As raízes negativas podem ser obtidas pela simetria dos polinômios de Legendre, dado que P_N(-x) = (-1)^N P_N(x).

    Retorna duas listas de arrays. Uma lista "raizes" em que cada linha N corresponde a um array com as raízes
    não negativas de P_N e uma lista "pesos" em que cada linha corresponde a um array com os respectivos pesos.

    Se imprimir == True, imprime os elementos das duas listas."""

    # Consistências para as entradas da função:
    if not isinstance(grau, int):
        raise TypeError("O grau do polinômio deve ser um inteiro.")
    if grau < 1 or grau > 30:
        raise ValueError("O grau do polinômio deve ser um inteiro entre 1 e 30.")
    if imprimir not in (True, False):
        raise TypeError("O parâmetro 'imprimir' deve ser um booleano.")

    # Lista com as raízes não negativas dos polinômios
    raizes = [np.array([]), np.array([0.0])]  # P_0(x) = 1 e P_1(x) = x

    # Lista com os pesos omega_j dos polinômios para os nós não negativos
    pesos = [np.array([]), np.array([2.0])]

    # Caso trivial para grau 1
    if grau == 1:
        return raizes, pesos

    for n in range(2, grau + 1):
        raizes_n = []  # Guarda os valores calculados para as raízes de P_n
        pesos_n = []  # Guarda os valores calculados para os pesos de P_n

        # Polinômio P_n(x) e derivada P_n'(x)
        p_atual = lambda x, grau1=n: avaliar_legendre(grau1, x)[0]
        dp_atual = lambda x, grau1=n: avaliar_legendre(grau1, x)[1]

        # As raízes do polinômio P_n estão entrelaçadas entre as raízes do polinômio P_n-1 dentro do intervalo [-1, 1].
        # A busca por raízes não negativas de P_n será feita no intervalo entre 0 e a primeira raiz, nos intervalos entre
        # duas raízes consecutivas positivas de P_n-1 e no intervalo entre a última raiz e 1. Cada intervalo possui uma única raiz.
        raizes_anteriores = raizes[n - 1]
        intervalos = list(raizes_anteriores) + [1.0]

        # Se n é ímpar, x=0 é uma raiz de P_n
        if n % 2 == 1:
            # Acrescenta a raiz x=0 e calcula o peso correspondente
            raizes_n.append(0.0)
            derivada_em_zero = dp_atual(0.0)
            peso = 2.0 / (derivada_em_zero**2)
            pesos_n.append(peso)

        for i in range(len(intervalos) - 1):
            a = intervalos[i]
            b = intervalos[i + 1]

            # Chamada da função para obter as raizes positivas de P_n no intervalo [a,b] pelo Método de Newton Modificado
            resultado = zero_funcao(
                f=p_atual, df=dp_atual, a=a, b=b, x0=a, relatorio=False
            )
            raiz = resultado.x

            # Acrescenta valor da raiz calculada
            raizes_n.append(raiz)

            # Calcula o peso correspondente à raiz encontrada
            deriv_raiz = dp_atual(raiz)
            if np.isclose(deriv_raiz, 0.0, atol=np.finfo(float).eps, rtol=0.0):
                raise ZeroDivisionError("Derivada nula na raiz encontrada.")
            peso = 2.0 / ((1.0 - raiz**2) * (deriv_raiz**2))
            pesos_n.append(peso)

        # Ao terminar de achar todas as raízes para o grau N:
        raizes.append(np.array(raizes_n))
        pesos.append(np.array(pesos_n))

    # Imprime os valores calculados pela função
    if imprimir:
        print(f"{'Grau P_N':<9} | {'Raízes':<65} | {'Pesos'}")
        print("-" * 110)
        for n in range(1, len(raizes)):
            rn = raizes[n]
            wn = pesos[n]

            if len(rn) > 0:
                str_raizes = ", ".join([f"{x:.4f}" for x in rn])
                str_pesos = ", ".join([f"{w:.4f}" for w in wn])

                print(f"{n:<9} | {str_raizes:<65} | {str_pesos}")

    return raizes, pesos


# TAREFA 3.1: Determinar k para a queda de um corpo com velocidade alvo v_alvo
# # Resultado
encontrar_k_queda_corpo(relatorio=True)

# TAREFA 3.2 Altura dos FIos de transmissao de eletrecidade
# # Resultado
encontrar_beta_catenaria(relatorio=True)


# TAREFA 4: Formulas de quadratura de Gauss-Legendre
# Resultado
print("\n\nTAREFA 4:\n")
calcula_valores_quadratura(grau=16, imprimir=True)

# @ Roberta
