#!/bin/bash

for i in 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016
#for i in 2015 2016
do
    python3 CompareResult.py $i --SP1500
done
