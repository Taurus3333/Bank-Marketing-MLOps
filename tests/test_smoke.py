from bank_marketing.core.logging import get_logger
from bank_marketing.core.exceptions import AppError

def test_logger_and_exception_work():
    log = get_logger("smoke")
    log.info("smoke: logger online")
    try:
        raise AppError("boom", details={"phase": "smoke"})
    except AppError as e:
        assert "boom" in str(e)
