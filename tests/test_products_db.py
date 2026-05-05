# from pathlib import Path

# import pytest

# from products_db import init_db, list_all_products, search_products


# @pytest.fixture
# def db_path(tmp_path: Path) -> Path:
#     path = tmp_path / "products.db"
#     init_db(path)
#     return path


# def test_init_seeds_products_on_empty_db(db_path: Path) -> None:
#     products = list_all_products(db_path)
#     assert len(products) >= 5
#     assert all(p.price_baht > 0 for p in products)
#     assert all(p.name for p in products)


# def test_init_is_idempotent(db_path: Path) -> None:
#     before = list_all_products(db_path)
#     init_db(db_path)
#     init_db(db_path)
#     after = list_all_products(db_path)
#     assert len(before) == len(after)


# def test_search_finds_by_name_case_insensitive(db_path: Path) -> None:
#     results = search_products(db_path, "headphones")
#     assert any("Headphones" in p.name for p in results)

#     results_upper = search_products(db_path, "HEADPHONES")
#     assert any("Headphones" in p.name for p in results_upper)


# def test_search_finds_by_description(db_path: Path) -> None:
#     results = search_products(db_path, "bluetooth")
#     assert len(results) >= 1
#     assert all(
#         "bluetooth" in p.name.lower() or "bluetooth" in p.description.lower()
#         for p in results
#     )


# def test_search_returns_empty_for_no_match(db_path: Path) -> None:
#     assert search_products(db_path, "definitely-not-a-product-xyz") == []


# def test_search_respects_limit(db_path: Path) -> None:
#     results = search_products(db_path, "", limit=2)
#     assert len(results) <= 2


# def test_in_stock_items_sort_first(db_path: Path) -> None:
#     products = list_all_products(db_path)
#     in_stock_indices = [i for i, p in enumerate(products) if p.in_stock]
#     out_of_stock_indices = [i for i, p in enumerate(products) if not p.in_stock]
#     if in_stock_indices and out_of_stock_indices:
#         assert max(in_stock_indices) < min(out_of_stock_indices)


# def test_product_summary_includes_price_and_stock(db_path: Path) -> None:
#     products = list_all_products(db_path)
#     summary = products[0].to_summary()
#     assert "baht" in summary
#     assert str(products[0].price_baht) in summary
