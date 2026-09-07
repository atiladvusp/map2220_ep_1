from __future__ import annotations

import math

import numpy as np
import pytest

from src.main import Resultado, zero_funcao


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


def test_derivada_muito_pequena_cai_para_dicotomia() -> None:
    f = lambda x: x**3 - 1.0e-12
    df = lambda x: 3.0 * x * x
    resultado = zero_funcao(f, df, 0.0, 1.0, x0=0.0)
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
    f = lambda x: x * x - 2.0
    df = lambda x: 2.0 * x
    resultado = zero_funcao(f, df, 0.0, 2.0, atol=1.0e-16, rtol=1.0e-16, maxit=1)
    assert resultado.convergiu is False
    assert resultado.motivo_parada == "maxit"
    assert resultado.iteracoes == 1


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
