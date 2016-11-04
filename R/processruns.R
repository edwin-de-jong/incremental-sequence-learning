processruns <- function(exp, mode, runnrlist, binsize_nrpoints, windowsize, folder, required_frac_available_timepoints = .8){
    deltarunnr=max(0,1-min(runnrlist))
    print(paste('deltarunnr',deltarunnr))

    if(F){
        exp='xc810t5'
        mode='train'
        runnrlist=1
        binsize_nrpoints=10000
        runnr=1
        required_frac_available_timepoints=.9
        windowsize=100
        folder = '~/code/digits/rnn'

    }

    library(zoo)

    result=NULL
    nmat=NULL
    meanmat=NULL
    M2mat=NULL
    nrpoints = NULL
    tablist=list()
    for (runnr in runnrlist){  
        print(paste('r',runnr))
        flush.console()

        print('before process')
        tab=process(paste(exp,'r',runnr,sep=''), mode, folder )								
        print('after process')


        tablist[[runnr+deltarunnr]]=tab$epoch
        if (is.null(result)){
            result=tab

            e1=as.matrix(result$epoch)
            nrmis = apply(is.na(e1),1,sum)
            sel = which(nrmis==0)
            e1 = e1[sel,]	
            
            xmat = result$epoch[ sel, "totnrpoints_trained" ]
            maxx = max( xmat, na.rm=T )
            xrange = seq( 0, maxx, binsize_nrpoints )
            nrbins = length( xrange )
            print(paste('maxx',maxx,'binsize_nrpoints',binsize_nrpoints,'nrbins',nrbins))
            
            e1r = nrbins
            e1c = dim(e1)[2]
            print(paste('dims',e1r,e1c))
            e1new = matrix( 0, e1r, e1c )

            for( c in 1:e1c ){
                y = e1[ , c ]
                if ( is.factor( y ) )
                {
                    y = levels( y )[ y ]
                }
                y = as.numeric( y )
                                        
                z = rollmean(zoo(y), k = windowsize, fill = "extend", partial = T )
                zx = rollmean(zoo(xmat), k = windowsize, fill = "extend", partial = T )
                z = approx( zx, z, xout = xrange )$y

                e1new[ , c ] = z
            }

            nrmis = apply(is.na(e1new),1,sum)
            sel = which(nrmis==0)

            e1new = e1new[sel,]
            result$epoch=e1new

            nmat = (0 * e1new) + 1 #it 1 of https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
            delta= e1new
            meanmat=delta
            M2mat = e1new*e1new

            print(paste('dims nrpoints:',dim(result$epoch)[1]))
            nrpointsepoch = c( matrix( 1, dim( result$epoch)[1],1),  matrix( 0, dim( result$epoch )[1],1) )
            

        }
        else{
            
            e1=as.matrix(result$epoch)
            e2=as.matrix(tab$epoch)

            nrmis = apply( is.na( e2 ), 1, sum )
            sel = which( nrmis == 0 ) 
            e2 = e2[ sel, ]	

            e2r = nrbins
            e2c = dim(e2)[2]

            e1r = dim(e1)[1]
            e1c = dim(e1)[2]

            e2new=matrix(0,e2r,e2c)
            xmat = tab$epoch[ sel , "totnrpoints_trained" ]
            
            for( c in 1:e2c ){
                y = e2[ , c ]
                if ( is.factor( y ) )
                {
                    y = levels( y )[ y ]
                }
                y = as.numeric( y )
                                        
                z = rollmean(zoo(y), k = windowsize, fill = "extend", partial = T )
                zx = rollmean(zoo(xmat), k = windowsize, fill = "extend", partial = T )
                z = approx( zx, z, xout = xrange )$y

                e2new[ , c ] = z
            }

            nrmis = apply( is.na( e2new ), 1,sum )
            sel = which( nrmis == 0 )
            e2new = e2new[ sel, ]

            e2 = e2new
            e2r = dim( e2 )[1]
            e2c = dim( e2 )[2]    
            
            res = matrix( 0, max( e1r, e2r ), max( e1c, e2c ))    
            res[1:e1r,1:e1c]=e1
            res[1:e2r,1:e2c]=res[1:e2r,1:e2c]+e2

            
            nmatnew=matrix(0,max(e1r,e2r),max(e1c,e2c))    
            nmatnew[1:e1r,1:e1c]=nmat
            nmatnew[1:e2r,1:e2c]=nmatnew[1:e2r,1:e2c]+1
            nmat=nmatnew

            meanmatnew=matrix(0,max(e1r,e2r),max(e1c,e2c))    
            meanmatnew[1:e1r,1:e1c]=meanmat
                                        
            meanmat=meanmatnew

            delta = e2 - meanmat[1:e2r,1:e2c]

            meanmat[1:e2r,1:e2c] = meanmat[1:e2r,1:e2c] + delta / nmat[1:e2r,1:e2c]

            M2matnew=matrix(0,max(e1r,e2r),max(e1c,e2c))    
            M2matnew[1:e1r,1:e1c]=M2mat
            M2matnew[1:e2r,1:e2c]=M2matnew[1:e2r,1:e2c]+delta*(e2-meanmat[1:e2r,1:e2c])
            M2mat=M2matnew

            nrpointsepoch[1:e2r] = nrpointsepoch[1:e2r] + 1
            result$epoch = res

        }
    }

    nrr=dim(result$epoch)[1]
    nrc=dim(result$epoch)[2]
    nrp=matrix(0,nrr,nrc)
    for(i in 1:nrc){
        nrp[,i]=nrpointsepoch[1:nrr]	
    }

    sel=which(nrpointsepoch >= required_frac_available_timepoints * length(runnrlist))
    nrpointsepoch=nrpointsepoch[sel]

    avg = result$epoch[sel,] / nrp[sel,]
    colnames(avg)=colnames(tab$epoch)

    nmat[is.na(nmat)]=0
    sel = nmat >= 2
    std = NA * meanmat
    std[sel]=sqrt(M2mat[sel]/(nmat[sel]-1))
    colnames(std) = colnames(avg)

    result={}
    result$avg=avg
    result$std=std
    result$tabs=tablist
    result$nrpoints = nrpointsepoch
    return(result)
}
