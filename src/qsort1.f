        subroutine QSORT1(val,ind,stack,n)
        integer n, ind(n),stack(n)
        doubleprecision val(n)
c-----------------------------------------------------------------------
c     does a quick-sort of an integer array.
c     on input val(1:n), is a real array, ind(1:n) is an integer array
c     on output ind(1:n) is permuted such that its elements are in 
c               increasing order. val(1:n) is an real array which 
c               permuted in the same way as ind(*).
c
c    code taken from SPARSKIT of Yousef Saad.
c    adapted by Matthias Bollhoefer 
c
c     This program is free software: you can redistribute it and/or modify
c     it under the terms of the GNU General Public License as published by
c     the Free Software Foundation, either version 3 of the License, or
c     (at your option) any later version.
c
c     This program is distributed in the hope that it will be useful,
c     but WITHOUT ANY WARRANTY; without even the implied warranty of
c     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
c     GNU General Public License for more details.
c
c     You should have received a copy of the GNU General Public License
c     along with this program.  If not, see <https://www.gnu.org/licenses/>.
c
c
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
        doubleprecision valtmp
        integer indtmp, key, first, last,top, mid, j
c-----
        top=0
        first=1
        last =n
c
c     outer loop -- while first<last or top>0 do
c
 1      if (first.ge.last .and. top.eq.0) goto 999
        mid = first
        key = ind(mid)
        do 2 j=first+1, last
           if (ind(j) .lt. key) then
              mid = mid+1
c     interchange entries at position j and mid
              valtmp = val(mid)
              val(mid) = val(j)
              val(j)  = valtmp

              indtmp = ind(mid)
              ind(mid) = ind(j)
              ind(j) = indtmp
           endif
 2      continue
c
c     interchange
        valtmp = val(mid)
        val(mid) = val(first)
        val(first)  = valtmp
c
        indtmp = ind(mid)
        ind(mid) = ind(first)
        ind(first) = indtmp
c
c     test for while loop
        if (first.lt.mid-1 .or. last.gt.mid+1) then
           if (first .lt. mid-1) then
              top=top+1
              stack(top)=mid
              last=mid-1
           else
              first=mid+1
           endif
        else
           if (top.gt.0) then
              first=stack(top)+1
              stack(top)=0
              top=top-1
              if (top.gt.0) then
                 last=stack(top)-1
              else
                 last=n
              endif
           else
              first=last
           endif
        end if
        goto 1
 999    return
        end
c----------------end-of-qsort1------------------------------------------
c-----------------------------------------------------------------------
