from fastapi import APIRouter
from Apps.UseCases.KYC.Api.routers import mock_kyc_api, mock_sanctions_api, kyc_onboarding_api

router = APIRouter(tags=["KYC UseCase APIs"])

# Include active routers required for CKYC Onboarding
router.include_router(mock_kyc_api.router)
router.include_router(mock_sanctions_api.router)
router.include_router(kyc_onboarding_api.router)



