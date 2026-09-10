from __future__ import annotations

import math

import numpy as np
import pytest

from src.main import (
    Resultado,
    encontrar_beta_catenaria,
    encontrar_k_queda_corpo,
    zero_funcao,
)


def _assert_convergencia_basica(resultado: Resultado, raiz_esperada: float) -> None:
    assert resultado.convergiu is True
    assert abs(resultado.x - raiz_esperada) < 1.0e-8
    assert abs(resultado.fx) < 1.0e-8
    assert resultado.iteracoes >= 1


def test_zero_funcao_linear() -> None:
    f = lambda x: x - 2.0
    df = lambda x: 1.0
    resultado = zero_funcao(f, df, 0.0, 5.0)
    _assert_convergencia_basica(resultado, 2.0)


def test_zero_funcao_quadratica() -> None:
    f = lambda x: x * x - 2.0
    df = lambda x: 2.0 * x
    resultado = zero_funcao(f, df, 0.0, 2.0)
    _assert_convergencia_basica(resultado, math.sqrt(2.0))


def test_zero_funcao_cubica() -> None:
    f = lambda x: x**3 - x - 2.0
    df = lambda x: 3.0 * x * x - 1.0
    resultado = zero_funcao(f, df, 1.0, 2.0)
    _assert_convergencia_basica(resultado, 1.5213797068045676)


def test_zero_funcao_trigonometrica() -> None:
    f = lambda x: math.sin(x)
    df = lambda x: math.cos(x)
    resultado = zero_funcao(f, df, 3.0, 4.0)
    _assert_convergencia_basica(resultado, math.pi)


def test_zero_funcao_exponencial() -> None:
    f = lambda x: math.exp(x) - 3.0
    df = lambda x: math.exp(x)
    resultado = zero_funcao(f, df, 0.0, 2.0)
    _assert_convergencia_basica(resultado, math.log(3.0))


def test_forca_dicotomia_quando_newton_sai_do_intervalo() -> None:
    f = lambda x: math.exp(x) - 10.0
    df = lambda x: math.exp(x)
    resultado = zero_funcao(f, df, 0.0, 3.0)
    assert resultado.convergiu is True
    assert resultado.contagem_bissecao >= 1
    assert resultado.contagem_newton >= 1
    assert abs(resultado.x - math.log(10.0)) < 1.0e-8


def test_forca_dicotomia_quando_newton_sai_do_intervalo_2() -> None:
    # Testa com caso de arctan com raiz inicial distante do zero.
    f = lambda x: math.atan(x)
    df = lambda x: 1.0 / (1.0 + x * x)
    resultado = zero_funcao(f, df, -100.0, 1000.0, x0=999.0)
    assert resultado.convergiu is True
    assert resultado.contagem_bissecao >= 1
    assert resultado.contagem_newton >= 1
    assert abs(resultado.x - math.tan(0.0)) < 1.0e-8


def test_metodo_somente_newton_nunca_usa_dicotomia() -> None:
    # Mesmo caso de test_forca_dicotomia_quando_newton_sai_do_intervalo, onde
    # "modificado" usa tanto Newton quanto dicotomia.
    f = lambda x: math.exp(x) - 10.0
    df = lambda x: math.exp(x)
    resultado = zero_funcao(f, df, 0.0, 3.0, metodo="somente_newton")
    assert resultado.contagem_bissecao == 0
    assert set(resultado.historico_metodos) == {"newton"}
    assert resultado.convergiu is True
    assert abs(resultado.x - math.log(10.0)) < 1.0e-8


def test_metodo_somente_bissecao_nunca_usa_newton() -> None:
    # Mesmo caso de test_forca_dicotomia_quando_newton_sai_do_intervalo, onde
    # "modificado" usa tanto Newton quanto dicotomia.
    f = lambda x: math.exp(x) - 10.0
    df = lambda x: math.exp(x)
    resultado = zero_funcao(f, df, 0.0, 3.0, metodo="somente_bissecao")
    assert resultado.contagem_newton == 0
    assert set(resultado.historico_metodos) == {"dicotomia"}
    assert resultado.convergiu is True
    assert abs(resultado.x - math.log(10.0)) < 1.0e-7


def test_metodo_somente_bissecao_nao_chama_df() -> None:
    def df_lanca_erro(_x: float) -> float:
        raise AssertionError("df nao deveria ser chamada em somente_bissecao")

    f = lambda x: math.exp(x) - 10.0
    resultado = zero_funcao(f, df_lanca_erro, 0.0, 3.0, metodo="somente_bissecao")
    assert resultado.convergiu is True


def test_metodo_somente_newton_propaga_erro_de_derivada_nula() -> None:
    f = lambda x: x**3
    df = lambda x: 3.0 * x * x
    with pytest.raises(ZeroDivisionError):
        zero_funcao(f, df, -1.0, 1.0, x0=0.0, metodo="somente_newton")


def test_metodo_invalido_gera_erro() -> None:
    f = lambda x: x - 2.0
    df = lambda x: 1.0
    with pytest.raises(ValueError, match="metodo"):
        zero_funcao(f, df, 0.0, 5.0, metodo="invalido")


def test_funcao_pode_nao_convergir_so_newton_1() -> None:
    f = lambda x: x**3 - 2 * x + 2
    df = lambda x: 3 * x * x - 2
    resultado = zero_funcao(f, df, -2, 2, x0=0, metodo="modificado")
    assert resultado.convergiu is True
    resultado = zero_funcao(f, df, -2, 2, x0=0, metodo="somente_newton")
    assert resultado.convergiu is False
    assert resultado.motivo_parada == "maxit"


def test_funcao_pode_nao_convergir_so_newton_2() -> None:
    f = lambda x: x**3 - 3 * x
    df = lambda x: 3 * x**2 - 3
    resultado = zero_funcao(f, df, -2, 2, x0=1, metodo="modificado")
    assert resultado.convergiu and resultado.x == 0.0
    with pytest.raises(ZeroDivisionError):
        resultado = zero_funcao(f, df, -2, 2, x0=1, metodo="somente_newton")


def test_funcao_roberta() -> None:
    f = lambda x: x**3 - 3 * x + 1
    df = lambda x: 3 * x**2 - 3
    resultado = zero_funcao(f, df, -2, 0, x0=-1, metodo="modificado")
    assert resultado.convergiu and np.isclose(resultado.x, -1.8794)


def test_derivada_muito_pequena_cai_para_dicotomia() -> None:
    f = lambda x: x**3 - 1.0e-12
    df = lambda x: 3.0 * x * x
    resultado = zero_funcao(f, df, 0.0, 1.0, x0=1.0e-12, atol=1.0e-16)
    assert resultado.convergiu is True
    assert resultado.contagem_bissecao >= 1
    assert abs(resultado.x - 1.0e-4) < 1.0e-7


def test_raiz_proxima_de_zero_com_atol() -> None:
    f = lambda x: x - 1.0e-12
    df = lambda x: 1.0
    resultado = zero_funcao(f, df, -1.0, 1.0, atol=1.0e-9, rtol=0.0)
    assert resultado.convergiu is True
    assert abs(resultado.x - 1.0e-12) < 1.0e-9


def test_parada_por_maxit() -> None:
    f = lambda x: x**3 - 2 * x + 2
    df = lambda x: 3 * x * x - 2
    resultado = zero_funcao(f, df, -2, 2, x0=0, metodo="somente_newton", maxit=5)
    assert resultado.convergiu is False
    assert resultado.motivo_parada == "maxit"
    assert resultado.iteracoes == 5


def test_intervalo_sem_troca_de_sinal_gera_erro() -> None:
    f = lambda x: x * x + 1.0
    df = lambda x: 2.0 * x
    with pytest.raises(ValueError, match="troca de sinal"):
        zero_funcao(f, df, -1.0, 1.0)


def test_x0_invalido_gera_erro() -> None:
    f = lambda x: x - 2.0
    df = lambda x: 1.0
    with pytest.raises(ValueError, match="x0 deve estar dentro do intervalo"):
        zero_funcao(f, df, 0.0, 5.0, x0=-1.0)


def test_parametros_invalidos_geram_erro() -> None:
    f = lambda x: x - 1.0
    df = lambda x: 1.0
    with pytest.raises(ValueError):
        zero_funcao(f, df, 2.0, 1.0)
    with pytest.raises(ValueError):
        zero_funcao(f, df, 0.0, 2.0, atol=-1.0)
    with pytest.raises(ValueError):
        zero_funcao(f, df, 0.0, 2.0, rtol=-1.0)
    with pytest.raises(ValueError):
        zero_funcao(f, df, 0.0, 2.0, maxit=0)


def test_relatorio_em_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    f = lambda x: x - 2.0
    df = lambda x: 1.0
    resultado = zero_funcao(f, df, 0.0, 5.0, relatorio=True)
    capturado = capsys.readouterr()
    assert resultado.convergiu is True
    assert "Relatorio do metodo de Newton modificado" in capturado.out
    assert "Iteracoes realizadas" in capturado.out
    assert "Passos de Newton" in capturado.out
    assert "Passos de dicotomia" in capturado.out


def test_historico_consistente() -> None:
    f = lambda x: np.exp(x) - 10.0
    df = lambda x: np.exp(x)
    resultado = zero_funcao(f, df, 0.0, 3.0)
    assert len(resultado.historico_metodos) == resultado.iteracoes
    assert set(resultado.historico_metodos).issubset({"newton", "dicotomia"})


def test_encontrar_k_queda_corpo() -> None:
    resultado = encontrar_k_queda_corpo()
    assert resultado.convergiu is True
    M, g, v0, t = 1.0, 10.0, 3.0, 2.0
    k = resultado.x
    v_t = (M * g - math.exp(-(t / M) * k) * (M * g - v0 * k)) / k
    assert abs(v_t - 20.0) < 1.0e-7


def test_encontrar_beta_catenaria() -> None:
    resultado = encontrar_beta_catenaria()
    assert resultado.convergiu is True
    beta = resultado.x
    diferenca = beta * (math.cosh(10.0 / beta) - 1.0)
    assert abs(diferenca - 0.5) < 1.0e-7


from numpy.polynomial.legendre import Legendre, leggauss

from src.main import (
    avaliar_legendre,
    calcula_valores_quadratura,
)


def test_legendre_p3():
    x = 0.3

    p, dp = avaliar_legendre(3, x)

    p_exato = (5.0 * x**3 - 3.0 * x) / 2.0
    dp_exato = 1.5 * (5.0 * x**2 - 1.0)

    assert p == pytest.approx(p_exato)
    assert dp == pytest.approx(dp_exato)


def test_legendre_p4():
    x = 0.3

    p, dp = avaliar_legendre(4, x)

    p_exato = (35.0 * x**4 - 30.0 * x**2 + 3.0) / 8.0
    dp_exato = (35.0 * x**3 - 15.0 * x) / 2.0

    assert p == pytest.approx(p_exato)
    assert dp == pytest.approx(dp_exato)


@pytest.mark.parametrize("n", range(0, 11))
def test_legendre_em_x_igual_1(n):
    p, dp = avaliar_legendre(n, 1.0)

    assert p == pytest.approx(1.0)
    assert dp == pytest.approx(n * (n + 1.0) / 2.0)


@pytest.mark.parametrize("n", range(0, 11))
def test_legendre_em_x_igual_menos_1(n):
    p, dp = avaliar_legendre(n, -1.0)

    valor_esperado = (-1.0) ** n
    derivada_esperada = (-1.0) ** (n + 1) * n * (n + 1.0) / 2.0

    assert p == pytest.approx(valor_esperado)
    assert dp == pytest.approx(derivada_esperada)


@pytest.mark.parametrize("grau", [7, 10])
@pytest.mark.parametrize("x", [-0.75, -0.5, 0.0, 0.5, 0.75])
def test_avaliar_legendre(grau: int, x: float) -> None:
    """
    Testa a função avaliar_legendre para P_7(x) ao P_10(x) comparando o valor do
    polinômio e de sua derivada com a implementação de referência do NumPy.
    """
    # Valor obtido pela função avaliar_legendre
    val_obtido, deriv_obtida = avaliar_legendre(grau, x)

    # Valor esperado (calculado via NumPy)
    leg = Legendre.basis(grau)
    val_esperado = leg(x)
    deriv_esperada = leg.deriv()(x)

    # Validação com tolerância para erros de ponto flutuante
    assert val_obtido == pytest.approx(val_esperado, abs=1e-12)
    assert deriv_obtida == pytest.approx(deriv_esperada, abs=1e-12)


def test_quadratura_grau_2():
    raizes, pesos = calcula_valores_quadratura(2, imprimir=False)

    assert raizes[2] == pytest.approx([1.0 / np.sqrt(3.0)])

    assert pesos[2] == pytest.approx([1.0])


def test_quadratura_grau_3():
    raizes, pesos = calcula_valores_quadratura(3, imprimir=False)

    raizes_exatas = [
        0.0,
        np.sqrt(3.0 / 5.0),
    ]

    pesos_exatos = [
        8.0 / 9.0,
        5.0 / 9.0,
    ]

    assert raizes[3] == pytest.approx(raizes_exatas)
    assert pesos[3] == pytest.approx(pesos_exatos)


@pytest.mark.parametrize("n", range(1, 17))
def test_raizes_realmente_sao_raizes(n):
    raizes, _ = calcula_valores_quadratura(n, imprimir=False)

    for x in raizes[n]:
        p, _ = avaliar_legendre(n, x)
        assert p == pytest.approx(0.0, abs=1e-8)


def test_calcula_valores_quadratura_grau_7() -> None:
    """
    Testa a função calcula_valores_quadratura para N=7 comparando os nós não-negativos
    e seus respectivos pesos com a referência do NumPy.
    """
    grau = 7
    raizes, pesos = calcula_valores_quadratura(grau=grau, imprimir=False)

    # NumPy retorna os nós ordenados de forma crescente [-x_max, ..., +x_max]
    nos_ref, pesos_ref = leggauss(grau)

    # Seleciona as raízes não-negativas (as últimas (N+1)//2 raízes)
    inicio_nao_negativos = grau // 2
    nos_ref_pos = nos_ref[inicio_nao_negativos:]
    pesos_ref_pos = pesos_ref[inicio_nao_negativos:]

    # Se N for ímpar, força o nó central (que teoricamente é 0.0) a ser exatamente 0.0
    if grau % 2 != 0:
        nos_ref_pos[0] = 0.0

    # Validação dos resultados
    np.testing.assert_allclose(raizes[7], nos_ref_pos, atol=1e-10)
    np.testing.assert_allclose(pesos[7], pesos_ref_pos, atol=1e-10)


@pytest.mark.parametrize("n", range(1, 17))
def test_pesos_positivos(n):
    _, pesos = calcula_valores_quadratura(n, imprimir=False)

    assert np.all(pesos[n] > 0.0)


@pytest.mark.parametrize("n", range(1, 17))
def test_soma_dos_pesos(n):
    _, pesos = calcula_valores_quadratura(n, imprimir=False)

    if n % 2 == 0:
        soma = 2.0 * np.sum(pesos[n])
    else:
        soma = pesos[n][0] + 2.0 * np.sum(pesos[n][1:])

    assert soma == pytest.approx(2.0, abs=1e-9)


def test_integracao_polinomio_grau_10() -> None:
    """
    Testa a quadratura de Gauss-Legendre integrando f(x) = x^10 + 3*x^8 - 2*x^5 + x^2 + 1
    no intervalo [-1, 1] com n=6 nós.
    Uma quadratura com n nós é exata para polinômios de grau até 2n - 1 (2*6 - 1 = 11).
    """

    # Definição da função integranda e sua integral exata no intervalo [-1, 1]
    # f(x) = x^10 + 3*x^8 - 2*x^5 + x^2 + 1
    def f(x):
        return x**10 + 3.0 * x**8 - 2.0 * x**5 + x**2 + 1.0

    # Integral exata: \int_{-1}^{1} (x^10 + 3x^8 - 2x^5 + x^2 + 1) dx = 2/11 + 6/9 + 0 + 2/3 + 2
    integral_exata = (
        (2.0 / 11.0) + (6.0 / 9.0) + (2.0 / 3.0) + 2.0
    )  # = 116 / 33 ~ 3.515151...

    # Obtém raízes e pesos não-negativos para n = 6
    raizes, pesos = calcula_valores_quadratura(grau=6, imprimir=False)
    nos_positivos = raizes[6]
    pesos_positivos = pesos[6]

    # Reconstrução da regra de quadratura usando a simetria de Gauss-Legendre:
    # \sum w_i * f(x_i) = \sum w_j * (f(x_j) + f(-x_j)) para x_j > 0
    integral_numerica = 0.0
    for x_j, w_j in zip(nos_positivos, pesos_positivos):
        if np.isclose(x_j, 0.0):
            integral_numerica += w_j * f(0.0)
        else:
            integral_numerica += w_j * (f(x_j) + f(-x_j))

    # O erro deve ser limitado apenas à precisão de ponto flutuante (~1e-10)
    assert integral_numerica == pytest.approx(integral_exata, abs=1e-10)
