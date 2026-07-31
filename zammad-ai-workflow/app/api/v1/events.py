"""Authenticated event ingress for standard Zammad webhooks."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.action.service import ActionService
from app.errors import AppError
from app.models.api_v1 import ZammadTicketEventInput, ZammadTicketEventOutput
from app.triage.triage import TriageService
from app.utils.logging import getLogger

from .answer import action_dependency
from .errors import app_error_to_http, unexpected_error_to_http
from .triage import triage_dependency
from .utils import check_api_key

logger = getLogger("zammad-ai.api.v1.events")
header_scheme = HTTPBearer(auto_error=False)

events_router = APIRouter(tags=["events"], prefix="/events")


@events_router.post(path="/zammad", status_code=status.HTTP_200_OK)
async def process_zammad_event(
    input: ZammadTicketEventInput,
    triage_service: TriageService = Depends(triage_dependency),
    action_service: ActionService = Depends(action_dependency),
    credentials: HTTPAuthorizationCredentials | None = Depends(header_scheme),
) -> ZammadTicketEventOutput:
    """Process one public customer article emitted by a Zammad trigger."""
    if not check_api_key(credentials):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        ticket = await triage_service.zammad_client.get_ticket(input.ticket_id)
        article = next((item for item in ticket.articles if item.id == input.article_id), None)
        if article is None:
            raise HTTPException(status_code=404, detail="Triggering article was not found on the ticket")
        if article.internal or (article.sender or "").casefold() != "customer":
            logger.info(
                f"Ignored non-customer article {input.article_id} on ticket {input.ticket_id}"
            )
            return ZammadTicketEventOutput(status="ignored", reason="not a public customer article")

        triage_result = await triage_service.perform_triage(ticket=ticket)
        await action_service.execute_action(ticket_id=input.ticket_id, triage=triage_result)
        return ZammadTicketEventOutput(status="processed")
    except HTTPException:
        raise
    except AppError as error:
        raise app_error_to_http(error) from error
    except Exception as error:
        logger.error("Zammad event processing failed", exc_info=True)
        raise unexpected_error_to_http() from error
