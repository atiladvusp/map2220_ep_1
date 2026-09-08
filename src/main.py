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
        else:
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
        delta = xn1 - xn
        if abs(delta) < atol + rtol * abs(xn1):
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
        if np.isclose(fxn1, 0.0, atol=np.finfo(float).eps, rtol=0.0):
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
    a: float = -1.0,
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


# TAREFA 3.1: Determinar k para a queda de um corpo com velocidade alvo v_alvo
# # Resultado
encontrar_k_queda_corpo(relatorio=True)


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


# TAREFA 3.2 Altura dos FIos de transmissao de eletrecidade
# # Resultado
encontrar_beta_catenaria(relatorio=True)


# TAREFA 4: Formulas de quadratura de Gauss-Legendre
# Resultado

# TAREFA 4: Formulas de quadratura de Gauss-Legendre
# Resultado
# @ Roberta
