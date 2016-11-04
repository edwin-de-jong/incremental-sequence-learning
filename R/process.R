
process <- function(exp, mode, folder){

    if(F){
        exp='xc810at5r1'
        mode='train'
        folder = '~/code/digits/rnn'
    }
    
    command = paste('cd ', folder ,';./processresults out/',exp,'.txt',sep='')
    print(command)
    system(command)

    fn = paste('./out/',exp,'-',mode,sep='');
    print(paste('reading ', fn))
    epoch=read.csv(fn,sep='',row.names=NULL)

                                        #remove columns with too many missing values
    nrmis = apply(is.na( epoch ),2,sum)
    nrrows=dim(epoch)[1]
    sel=nrmis < .5 * nrrows
    epoch = epoch[,sel]

    if (colnames(epoch)[1]=='row.names')
    {
        colnames(epoch)=c(colnames(epoch)[-1],'x')
    }

    rmse_own_output_0=NULL
    rmse_own_output_1=NULL
    
    fn = paste('./results/',exp,'/rmse-own-output0.txt',sep='')
    if (file.exists(fn)){
        rmse_own_output_0 = tryCatch({
            print(paste('reading ', fn))	
            read.table(fn,sep='')
        }, error = function(e) {
            print(paste('error reading file',e))
        })
    }
    
    fn = paste('./results/',exp,'/rmse-own-output1.txt',sep='')
    if (file.exists(fn)){
        rmse_own_output_1 = tryCatch({
            print(paste('reading ', fn))
            read.table(fn,sep='')
        }, error = function(e) {
            print(paste('error reading file',e))
        })
    }

    result={}
    result$epoch=epoch
    result$rmse_own_output_0=rmse_own_output_0
    result$rmse_own_output_1=rmse_own_output_1
    
    return(result)
}
