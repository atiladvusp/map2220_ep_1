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

    # Roberta, este dataclass eh com se fosse uma "caixa" apara agrupar os dados e
    # facilitar o testes e ate imprimir os resutlados
    # Para criar uma, baste instaconicar com os valores: exemplo:
    # resultado = Resultado(
    #     x=0.0,
    #     fx=0.0,
    #     iteracoes=0,
    #     convergiu=False,
    #     motivo_parada="",
    #     contagem_newton=0,
    #     contagem_bissecao=0,
    #     historico_metodos=[]
    # )

    x: float
    fx: float
    iteracoes: int
    convergiu: bool
    motivo_parada: str
    contagem_newton: int
    contagem_bissecao: int
    historico_metodos: list[str]


# Roberta, como falamos ontem, voce pode implementar a funcao que encontra razizes abaixo.
# Eu ja montei os testes levando em conisderacao a assinatura de funcao abaixo, mas voce
# pode modificar se achar necessario. NOte que o retorno da funcaoe eh o dataclass acima


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
    # @ Roberta
    return None
