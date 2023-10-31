
module mod_ftorch
   integer :: neededsize 
   integer, parameter :: arrsize = 78374
   double precision :: nn_input(arrsize,6), nn_output(arrsize,1)
end module mod_ftorch


! program to test this interface
program main
use mod_ftorch
implicit none
integer :: i,ipts
real(8) :: t1,t2,t3,t4

    open(76,file='torch_in.csv',form='formatted',status='old')
    do ipts=1, arrsize
      read(76,*) nn_input(ipts,1:6)  
    end do
    close(76)

! 1. using ftorch interface

    call cpu_time(t1)

    call ftorch_interface
    !output is store in nn_output

    call cpu_time(t2)

! 2. using system call

    call cpu_time(t3)
    call system('python run_pytorch.py')
!    open(75,file='torch_out.csv',form='formatted')
!    do ipts=1, arrsize
!      !copy output to nn_output for later use
!      read(75,*) nn_output(ipts,1)
!    end do
!    close(75,status='delete')
    call cpu_time(t4)

    print *,'ftorch timing',     t2-t1
    print *,'system call timing',t4-t3

end program main



subroutine ftorch_interface

! Import any C bindings as required for this code
use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_loc
! Import library for interfacing with PyTorch
use ftorch
! interface module for this subroutine
use mod_ftorch

implicit none

! Generate an object to hold the Torch model
type(torch_module) :: model

! Set up types of input and output data and the interface with C
integer(c_int), parameter :: dims_input = 2
integer(c_int64_t)        :: shape_input(dims_input)
integer(c_int), parameter :: n_inputs = 1 
type(torch_tensor), dimension(n_inputs) :: model_input_arr
integer(c_int), parameter :: dims_output = 2
integer(c_int64_t)        :: shape_output(dims_output)
type(torch_tensor)        :: model_output
integer(c_int)            :: layout_1(dims_input), layout_o(dims_output)

integer :: i
! Set up the model inputs as Fortran arrays
real, dimension(arrsize,6),   target  :: input  
real, dimension(arrsize,1),   target  :: output
double precision, dimension(6) :: mean, vari
logical :: first=.true.
real(8) :: time(8)
save

call cpu_time(time(1))
! Initialise the Torch model to be used
! This is the trained network model exported in torch script.
if(first) then
  model = torch_module_load("model_script.pt")

! Preprocessing of the input array
!!handing over input arrays and mean/variance arrays
!  mean(:) = (/-3.298436769915389d0,   2.5308645104000886d0,  0.9999999837223219d0, &
!               1.15349548097465d0,    1.5732733326927042d0,  2.003519347548962d0/)
!  vari(:) = (/5.8769289957979796D+00, 3.8224696430436542D+00, 6.0689705617948579D-17, &
!              1.8613085303547325D-01, 2.5796750878974462D-01, 3.0504456862273172D-01/)
  first = .false.
!  print *,'this should be done once'
end if
!
!! fit.transform()
!do i=1,arrsize
!   nn_input(i,:) = ( nn_input(i,:) - mean(:) ) /sqrt(vari(:))
!end do

input = nn_input

! Not sure what these layouts are, but they are needed.
layout_1 = (/ 1, 2 /)
layout_o = (/ 2, 1 /)

call cpu_time(time(2))

! Wrap Fortran data as no-copy Torch Tensors
! There may well be some reshaping required depending on the 
! structure of the model which is not covered here (see examples)
shape_input  = (/arrsize, 6/)
shape_output = (/arrsize, 1/)
model_input_arr(1) = torch_tensor_from_blob(c_loc(input),   dims_input,  shape_input,  torch_kFloat32, torch_kCPU, layout_1)
model_output       = torch_tensor_from_blob(c_loc(output),  dims_output, shape_output, torch_kFloat32, torch_kCPU, layout_o)

call cpu_time(time(3))
! Run model and Infer
call torch_module_forward(model, model_input_arr, n_inputs, model_output)
call cpu_time(time(4))

! Write out the result of running the model
!write(*,*) output

nn_output = output

! Clean up
! Run this when you are done
call torch_module_delete(model)
call torch_tensor_delete(model_input_arr(1))
call torch_tensor_delete(model_output)

call cpu_time(time(5))

print *,'nntiming ', time(2)-time(1)
print *,'nntiming ', time(3)-time(2)
print *,'nntiming ', time(4)-time(3)
print *,'nntiming ', time(5)-time(4)

end subroutine ftorch_interface
