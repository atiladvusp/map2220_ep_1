from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

FuncaoEscalar = Callable[[float], float]


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


def _criterio_intervalo_newton(
    xn: float,
    alpha: float,
    beta: float,
    fxn: float,
    dfxn: float,
) -> bool:
    termo_a = (xn - alpha) * dfxn - fxn
    termo_b = (xn - beta) * dfxn - fxn
    return termo_a * termo_b < 0.0


def _criterio_reducao_passo(
    fxn: float,
    dfxn: float,
    delta_xn: float | None,
) -> bool:
    if delta_xn is None:
        return True
    return 2.0 * abs(fxn) < abs(dfxn * delta_xn)


def _imprimir_relatorio(
    resultado: Resultado,
    a: float,
    b: float,
    atol: float,
    rtol: float,
    maxit: int,
) -> None:
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


def zero_funcao(
    f: FuncaoEscalar,
    df: FuncaoEscalar,
    a: float,
    b: float,
    x0: float | None = None,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-10,
    maxit: int = 100,
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
        relatorio: Se True, imprime um relatorio simples da execucao no stdout.

    Returns:
        Um objeto Resultado contendo a aproximacao final, o valor de f na
        aproximacao, o numero de iteracoes, o status de convergencia e o
        historico de metodos usados.

    Raises:
        TypeError: Se f ou df nao forem chamaveis.
        ValueError: Se o intervalo for invalido, se as tolerancias forem
            negativas, se maxit nao for positivo, se nao houver troca de sinal
            no intervalo ou se x0 nao for um dos extremos do intervalo.
    """
    if not callable(f) or not callable(df):
        raise TypeError("f e df devem ser funcoes chamaveis.")
    if not (a < b):
        raise ValueError("O intervalo deve satisfazer a < b.")
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("As tolerancias atol e rtol devem ser nao negativas.")
    if maxit <= 0:
        raise ValueError("maxit deve ser positivo.")

    alpha = float(a)
    beta = float(b)
    fa = float(f(alpha))
    fb = float(f(beta))

    if np.isclose(fa, 0.0, atol=np.finfo(float).eps, rtol=0.0):
        resultado = Resultado(
            x=alpha,
            fx=fa,
            iteracoes=0,
            convergiu=True,
            motivo_parada="raiz_exata",
            contagem_newton=0,
            contagem_bissecao=0,
            historico_metodos=[],
        )
        if relatorio:
            _imprimir_relatorio(resultado, a, b, atol, rtol, maxit)
        return resultado

    if np.isclose(fb, 0.0, atol=np.finfo(float).eps, rtol=0.0):
        resultado = Resultado(
            x=beta,
            fx=fb,
            iteracoes=0,
            convergiu=True,
            motivo_parada="raiz_exata",
            contagem_newton=0,
            contagem_bissecao=0,
            historico_metodos=[],
        )
        if relatorio:
            _imprimir_relatorio(resultado, a, b, atol, rtol, maxit)
        return resultado

    if fa * fb > 0.0:
        raise ValueError("O intervalo inicial deve ter troca de sinal: f(a)*f(b) < 0.")

    if x0 is None:
        xn = alpha if abs(fa) <= abs(fb) else beta
    else:
        xn = float(x0)
        if not (np.isclose(xn, alpha) or np.isclose(xn, beta)):
            raise ValueError("x0 deve ser um dos extremos do intervalo inicial.")

    # Mantemos os valores de f(alpha) e f(beta) sincronizados para evitar
    # recomputacoes desnecessarias e manter o metodo previsivel numericamente.
    f_alpha = fa
    f_beta = fb
    fxn = float(f(xn))

    if np.isclose(fxn, 0.0, atol=np.finfo(float).eps, rtol=0.0):
        resultado = Resultado(
            x=xn,
            fx=fxn,
            iteracoes=0,
            convergiu=True,
            motivo_parada="raiz_exata",
            contagem_newton=0,
            contagem_bissecao=0,
            historico_metodos=[],
        )
        if relatorio:
            _imprimir_relatorio(resultado, a, b, atol, rtol, maxit)
        return resultado

    eps_derivada = 1.0e-14
    iteracoes = 0
    convergiu = False
    motivo_parada = "maxit"
    contagem_newton = 0
    contagem_bissecao = 0
    historico_metodos: list[str] = []
    delta_xn: float | None = None

    for n in range(maxit):
        dfxn = float(df(xn))
        pode_newton = abs(dfxn) > eps_derivada
        criterio_2 = False
        criterio_3 = False

        if pode_newton:
            criterio_2 = _criterio_intervalo_newton(xn, alpha, beta, fxn, dfxn)
            criterio_3 = _criterio_reducao_passo(fxn, dfxn, delta_xn)

        if pode_newton and criterio_2 and criterio_3:
            xn1 = xn - fxn / dfxn
            metodo = "newton"
            contagem_newton += 1
        else:
            xn1 = 0.5 * (alpha + beta)
            metodo = "dicotomia"
            contagem_bissecao += 1

        historico_metodos.append(metodo)
        fxn1 = float(f(xn1))
        iteracoes = n + 1

        if np.isclose(fxn1, 0.0, atol=np.finfo(float).eps, rtol=0.0):
            xn = xn1
            fxn = fxn1
            convergiu = True
            motivo_parada = "raiz_exata"
            break

        # Atualizacao do intervalo isolante pela troca de sinal.
        if f_alpha * fxn1 < 0.0:
            beta = xn1
            f_beta = fxn1
        else:
            alpha = xn1
            f_alpha = fxn1

        delta = xn1 - xn
        if abs(delta) < atol + rtol * abs(xn1):
            xn = xn1
            fxn = fxn1
            convergiu = True
            motivo_parada = "tolerancia"
            break

        xn = xn1
        fxn = fxn1
        delta_xn = delta

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
        _imprimir_relatorio(resultado, a, b, atol, rtol, maxit)

    return resultado
