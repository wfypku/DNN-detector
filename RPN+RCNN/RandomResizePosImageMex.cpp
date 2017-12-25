#include "mex.h"
#include <math.h>
#include <string.h>
#include <algorithm>

template <typename T>
        void Bilinear(const T *pInput, int iWidth, int iHeight, int iChannels,
        float fX, float fY, T *pOutput, int iStrideOutput)
{
    // out of range
    if (fX < 0.f || fX >(float)(iWidth - 1) || fY < 0 || fY >(float)(iHeight - 1))
    {
        memset(pOutput, 0, sizeof(T)*iChannels);
        return;
    }
    
    // int pixels
    if (fX == (float)(int)(fX) && fY == (float)(int)(fY))
    {
        const T *p = pInput + (int)fX * iHeight + (int)fY;
        int s = iHeight * iWidth;
        for (int i = 0; i < iChannels; i++)
            pOutput[i*iStrideOutput] = p[i*s];
        return;
    }
    
    // otherwise
    int iLeft = (int)floor(fX);
    int iRight = iLeft + 1;
    int iTop = (int)floor(fY);
    int iBottom = iTop + 1;
    
    float dX0 = fX - iLeft;
    float dX1 = iRight - fX;
    float dY0 = fY - iTop;
    float dY1 = iBottom - fY;
    
    iRight = iRight >= iWidth ? iWidth - 1 : iRight;
    iBottom = iBottom >= iHeight ? iHeight - 1 : iBottom;
    
    const T *v00 = pInput + iTop + iLeft*iHeight;
    const T *v01 = pInput + iTop + iRight*iHeight;
    const T *v10 = pInput + iBottom + iLeft*iHeight;
    const T *v11 = pInput + iBottom + iRight*iHeight;
    const int s = iHeight * iWidth;
    for (int i = 0; i < iChannels; i++)
        pOutput[i*iStrideOutput] = (T)(dX1*dY1*v00[i*s] + dX1*dY0*v10[i*s] + dX0*dY0*v11[i*s] + dX0*dY1*v01[i*s]);
}


void RandomResizePosImage(const unsigned char *pImage, int iWidth, int iHeight, int iChannel,
        double fFaceLeft, double fFaceRight, double fFaceTop, double fFaceBottom,
        const double *pfPoints, int iPointsNum,
        double fMinFaceSize, double fMaxFaceSize, double fRandOffset,
        double fAnchorCenter, int iMinImageSize, int iMaxImageSize,
        mxArray* &pImageOutput,
        double &fFaceLeftOutput, double &fFaceRightOutput, double &fFaceTopOutput, double &fFaceBottomOutput,
        double *pfPointsOutput)
{
    double fRandFaceSize = fMinFaceSize * pow(2, log(fMaxFaceSize/fMinFaceSize)/log(2.0)*((double)rand()/(double)RAND_MAX));
    double fInputFaceSize = ((fFaceRight - fFaceLeft) + (fFaceBottom - fFaceTop))/2.0;
    double fResizeScale = fRandFaceSize / fInputFaceSize;
    int iCropStartX = 0, iCropStartY = 0, iCropEndX = int(iWidth*fResizeScale + 0.5), iCropEndY = int(iHeight*fResizeScale+0.5);
    double fResizeFaceCenterX = (fFaceLeft + fFaceRight) * fResizeScale / 2.0 + fRandOffset * ((double)rand()/(double)RAND_MAX*2.0 - 1.0);
    double fResizeFaceCenterY = (fFaceTop + fFaceBottom) * fResizeScale / 2.0 + fRandOffset * ((double)rand()/(double)RAND_MAX*2.0 - 1.0);
    if (iCropEndX - iCropStartX < iMinImageSize)
    {
        iCropStartX -= (iMinImageSize - (iCropEndX - iCropStartX))/2;
        iCropEndX = iCropStartX + iMinImageSize;
    }
    else if (iCropEndX - iCropStartX > iMaxImageSize)
    {
		iCropStartX = std::max(0, int(fResizeFaceCenterX - iMaxImageSize / 2 + 0.5));
		iCropEndX = std::min(iCropEndX, iCropStartX + iMaxImageSize);
		iCropStartX = iCropEndX - iMaxImageSize;
    }
	int iCurrentWidth = iCropEndX - iCropStartX;
	if (fResizeFaceCenterX - (double)iCropStartX < fAnchorCenter)
	{
		iCropStartX = (int)floor(fResizeFaceCenterX - fAnchorCenter);
		iCropEndX = iCropStartX + iCurrentWidth;
	}
	else if (fResizeFaceCenterX >(double)iCropEndX - fAnchorCenter - 1)
	{
		iCropEndX = (int)ceil(fResizeFaceCenterX + fAnchorCenter + 1);
		iCropStartX = iCropEndX - iCurrentWidth;
	}
    
	if (iCropEndY - iCropStartY < iMinImageSize)
	{
		iCropStartY -= (iMinImageSize - (iCropEndY - iCropStartY)) / 2;
		iCropEndY = iCropStartY + iMinImageSize;
	}
	else if (iCropEndY - iCropStartY > iMaxImageSize)
	{
		iCropStartY = std::max(0, int(fResizeFaceCenterY - iMaxImageSize / 2 + 0.5));
		iCropEndY = std::min(iCropEndY, iCropStartY + iMaxImageSize);
		iCropStartY = iCropEndY - iMaxImageSize;
	}
	int iCurrentHeight = iCropEndY - iCropStartY;
	if (fResizeFaceCenterY - (double)iCropStartY < fAnchorCenter)
	{
		iCropStartY = (int)floor(fResizeFaceCenterY - fAnchorCenter);
		iCropEndY = iCropStartY + iCurrentHeight;
	}
	else if (fResizeFaceCenterY >(double)iCropEndY - fAnchorCenter - 1)
	{
		iCropEndY = (int)ceil(fResizeFaceCenterY + fAnchorCenter + 1);
		iCropStartY = iCropEndY - iCurrentWidth;
	}
    
    
    int iOutputImageWidth = iCropEndX - iCropStartX;
    int iOutputImageHeight = iCropEndY - iCropStartY;
    mwSize dims[3] = {iOutputImageHeight, iOutputImageWidth, iChannel};
    pImageOutput = mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL);
    unsigned char *pOutput = (unsigned char *)mxGetData(pImageOutput);
//#pragma omp parallel for
    for (int w = 0; w < iOutputImageWidth; w++)
    {
        double x =  (w + iCropStartX) / fResizeScale;
        for (int h = 0; h < iOutputImageHeight; h++)
        {
            unsigned char *po = pOutput + w * iOutputImageHeight + h;
            double y = (h + iCropStartY) / fResizeScale;
            Bilinear(pImage, iWidth, iHeight, iChannel, (float)x, (float)y, po, iOutputImageWidth*iOutputImageHeight);
        }
    }
	fFaceLeftOutput = fFaceLeft * fResizeScale - (double)iCropStartX;
	fFaceRightOutput = fFaceRight * fResizeScale - (double)iCropStartX;
	fFaceTopOutput = fFaceTop * fResizeScale - (double)iCropStartY;
	fFaceBottomOutput = fFaceBottom * fResizeScale - (double)iCropStartY;
	for (int i = 0; i < iPointsNum; i++)
	{
		pfPointsOutput[2 * i] = pfPoints[2 * i] * fResizeScale - (double)iCropStartX;
		pfPointsOutput[2 * i + 1] = pfPoints[2 * i + 1] * fResizeScale - (double)iCropStartY;
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nlhs != 3 || nrhs != 4)
    {
        mexErrMsgTxt("[img_out, rect_out, points_out] = RandomResizePosImage(img, rect, points, rpn_model)");
    }
    const unsigned char *pImage = (const unsigned char *)mxGetData(prhs[0]);
    const mwSize *dims = mxGetDimensions(prhs[0]);
    int iWidth = dims[1];
    int iHeight = dims[0];
    int iChannel = mxGetNumberOfDimensions(prhs[0]) > 2 ? dims[2] : 1;
    const double *pRect = (const double *)mxGetData(prhs[1]);
    const double *pPoints = (const double *)mxGetData(prhs[2]);
    int iPointsNum = mxGetN(prhs[2])/2;
    mxArray *param = mxGetField(prhs[3], 0, "param");
    double fMinFaceSize = mxGetScalar(mxGetField(param, 0, "min_face_size"));
    double fMaxFaceSize = mxGetScalar(mxGetField(param, 0, "max_face_size"));
    double fRandOffset = mxGetScalar(mxGetField(param, 0, "max_rand_offset"));
    double fAnchorCenter = mxGetPr(mxGetField(prhs[3], 0, "anchor_center"))[0];
    int iMinImageSize = (int)mxGetScalar(mxGetField(prhs[3], 0, "receptive_field_size"));
    int iMaxImageSize = (int)mxGetScalar(mxGetField(param, 0, "max_img_size"));
    plhs[1] = mxCreateDoubleMatrix(1, 4, mxREAL);
    double *pRectOutput = mxGetPr(plhs[1]);
    plhs[2] = mxCreateDoubleMatrix(1, iPointsNum*2, mxREAL);
    double *pPointsOutput = mxGetPr(plhs[2]);
    RandomResizePosImage(pImage, iWidth, iHeight, iChannel, 
            pRect[0], pRect[1], pRect[2], pRect[3],
        pPoints, iPointsNum,
        fMinFaceSize, fMaxFaceSize, fRandOffset,
        fAnchorCenter, iMinImageSize, iMaxImageSize,
        plhs[0],
        pRectOutput[0], pRectOutput[1], pRectOutput[2], pRectOutput[3],
        pPointsOutput);
}


